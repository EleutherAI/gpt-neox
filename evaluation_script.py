#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2021 Josh Levy-Kramer <josh@levykramer.co.uk>. All rights reserved.
# This file is based on code by the authors denoted below and has been modified from its original version.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#run it using the following command: ./deepy.py evaluation_script.py -d configs sampling.yml small.yml

import os
import time
import torch
import wandb
import result_records
import memorization_metric
import numpy as np
from megatron import mpu, print_rank_0
from megatron.data import data_utils
from megatron import text_generation_utils
from megatron import utils as megatron_utils 

MODEL_GLOBAL_BATCH_SIZE = 2048# Global batch size of trained model

def score(neox_args,model,data,token_size=64):
    '''Calculates the memorization metric for the given input tokens

    Memorization metric for input tokens is average NLL loss and accuracy
    between input tokens and generated logits

    Arguments:
        neox_args: an instance of NeoXArgs containing the configuration for evaluation
        model: Megatron model for evaluation
        data: a single batch of evaluation dataset for evaluation
        token_size: Number of tokens used for both prompt and evaluation
    '''
    if data is not None:
        data = [i[:token_size] for i in data] # Use half of tokens for prompting
    else:
        data = [[0 for i in range(token_size)] for j in range(neox_args.train_micro_batch_size_per_gpu)]
    
    torch.distributed.broadcast_object_list(data, src=mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())
    
    
    inputs = [i[:token_size//2] for i in data]

    model.module.clear_cache()
    
    logits, context_tokens = text_generation_utils.stream_tokens( # Generation
        neox_args = neox_args, 
        model = model, 
        context_tokens = inputs,
        maximum_tokens=token_size,
        recompute=neox_args.recompute
    )

    
    logits = logits[token_size//2:token_size]
    ground_truth = torch.tensor([i[token_size//2:token_size] for i in data])
    context_tokens = context_tokens[:,token_size//2:token_size].cpu()
    nll_avg = memorization_metric.memorization_nll(logits, ground_truth)
    accuracy_avg = memorization_metric.memorization_acc(context_tokens, ground_truth)
    return np.stack((nll_avg, accuracy_avg), axis=-1)


def main():

    # Initialization
    os.environ['TORCH_EXTENSIONS_DIR'] = '/fsx/home-orz/.cache/torch_extensions/py38_cu116'
    model, neox_args = megatron_utils.setup_for_inference_or_eval()

    total_iters = neox_args.iteration*MODEL_GLOBAL_BATCH_SIZE
    total_iters //= neox_args.train_micro_batch_size_per_gpu*mpu.get_data_parallel_world_size()
    total_iters += 1
    
    megatron_utils.print_rank_0(f"Total Iterations: {total_iters}")
    
    model.eval()
    results_path = f'memorization_results_{neox_args.wandb_group}' + '.csv'
    iteration_path = f'memorization_iteration_{neox_args.wandb_group}'
    token_size = neox_args.maximum_tokens

    # Create result_records and initialize wandb

    os.environ["WANDB_RESUME"] = "allow"
    node_rank = str(int(os.environ["OMPI_COMM_WORLD_RANK"])//int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"]))
    os.environ["WANDB_RUN_ID"] = f"sairam-{node_rank}-{neox_args.iteration}"

    is_src_rank = mpu.get_data_parallel_rank() == 0
    is_src_rank &= mpu.get_model_parallel_rank() == 0
    if is_src_rank:
        records = result_records.DataFrameCreator(results_path, restart=True) #store results on rank 0
    
    megatron_utils.init_wandb(neox_args)    

    # Set current iteration, and build datasets

    if os.path.exists(iteration_path):
        with open(iteration_path) as f:
            neox_args.update_value("iteration", int(f.read()))
    else:
        neox_args.update_value("iteration", 0)

    ds, _, _ = data_utils.build_train_valid_test_data_iterators(neox_args=neox_args)

    # Evaluation driver code

    megatron_utils.print_rank_0("Starting Evaluation")
    idx = neox_args.iteration*neox_args.train_micro_batch_size_per_gpu
    idx *= mpu.get_data_parallel_world_size()

    # Iteratating over the dataset

    t = time.time()
    iteration = neox_args.iteration
    megatron_utils.print_rank_0(f"Total eval iters: {total_iters}")

    while iteration < (total_iters):
        iteration+=1
        if ds is not None:
            batch = next(ds)['text'].cpu().numpy().tolist()
        else:
            batch = None

        memorization = torch.tensor(score(neox_args,model,batch,token_size)).cuda()

        is_src_rank = mpu.get_data_parallel_rank() == 0
        is_src_rank &= mpu.get_model_parallel_rank() == 0
        if is_src_rank:
            total_nll_loss = 0
            total_accuracy = 0
            for i in memorization:
                records.write(idx, i[0], i[1])
                total_nll_loss += i[0]
                total_accuracy += i[1]
                idx+=1
            wandb.log({
                'index':idx,
                'nll_loss':total_nll_loss/len(memorization),
                'accuracy':total_accuracy/len(memorization)
            })

        if mpu.get_model_parallel_rank() != 0:
            continue

        if mpu.get_data_parallel_rank() != 0:
            torch.distributed.isend(memorization, 0, group=mpu.get_data_parallel_group())
        else:
            for i in range(1,mpu.get_data_parallel_world_size()):
                torch.distributed.recv(
                    memorization,
                    src=i*mpu.get_model_parallel_world_size(),
                    group = mpu.get_data_parallel_group()
                )
                total_nll_loss = 0
                total_accuracy = 0
                for i in memorization:
                    records.write(idx, i[0], i[1])
                    total_nll_loss += i[0]
                    total_accuracy += i[1]
                    idx+=1
                wandb.log({
                    'index':idx,
                    'nll_loss':total_nll_loss/len(memorization),
                    'accuracy':total_accuracy/len(memorization)
                })
            
            records.commit()
            with open(iteration_path, 'w') as f:
                f.write(str(iteration))

        megatron_utils.print_rank_0(f"Generation took {time.time() - t:.3}s for iteration {iteration}")
        t = time.time()

    is_src_rank = mpu.get_data_parallel_rank() == 0
    is_src_rank &= mpu.get_model_parallel_rank() == 0   
    if is_src_rank:
        records.close()
    
    


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as err:
        import requests
        import datetime
        ts = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')+'UTC'
        resp = requests.get('http://169.254.169.254/latest/meta-data/instance-id')
        print(f'ERROR at {ts} on {resp.text} device: {type(err).__name__}: {err}', flush=True)
        raise err
