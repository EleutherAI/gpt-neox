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

from tqdm import tqdm
import torch
import tensorflow as tf
import numpy as np
import wandb
from memorization_metric import memorization_metric
import argparse
from result_records import TFrecordCreator
from threading import Thread
import queue
import torch.distributed as dist
from megatron.data.data_utils import build_train_valid_test_datasets
from megatron import mpu
from megatron.text_generation_utils import stream_tokens
from megatron.utils import print_rank_0, setup_for_inference_or_eval
import os
import time

class BatchedDataset:
    def __init__(self,batch_size,take_every,token_size):
        super().__init__()
        self.batch_size = batch_size
        self.take_every = take_every
        self.token_size = token_size
    def __iter__(self):
        ds, valid_ds, test_ds = build_train_valid_test_datasets(
                data_prefix="/mnt/ssd-1/data/pile/pile_text_document",
                data_impl="mmap",
                splits_string="949,50,1",
                train_valid_test_num_samples=[210604984, 0, 0],
                seq_length=2048,
                seed=1234,
                skip_warmup=True
            )
        tokens = []
        indicies = []
        val = 1
        idx = 0
        for doc in ds:
            i = doc['text']
            idx += 1
            if(idx%self.take_every != 0 or i.shape[0] < self.token_size):
                continue
            tokens.append(i[:self.token_size])
            indicies.append(idx)
            if(val%self.batch_size == 0):
                yield (np.asarray(tokens),indicies)
                indicies = []
                tokens = []
            val += 1



def score(neox_args,model,data,token_size=64):
    '''Calculates the memorization metric for the given input tokens
    '''
    
    inp_tensor = torch.tensor(data[:,:token_size])
    inp = inp_tensor[:,:token_size//2]
    res = stream_tokens(
        neox_args=neox_args, 
        model=model,
        context_tokens = inp.tolist(),
        maximum_tokens = token_size, 
        recompute = neox_args.recompute, 
        temperature = neox_args.temperature,
        top_k = neox_args.top_k, 
        top_p = neox_args.top_p
    )
    res = res[:,token_size//2:token_size,:].cpu().transpose(0,1)
    return memorization_metric(res,inp_tensor[:,token_size//2:token_size])



def main():
    BATCH_SIZE = 512
    RESULTS_PATH = '/home/mchorse/gpt-neox/memorization_results_neox_dense_small.tfrecords'
    TOKEN_SIZE = 64
    TAKE_EVERY = 50

    records = TFrecordCreator(RESULTS_PATH) #store results
    
    model, neox_args = setup_for_inference_or_eval()
    ds = iter(BatchedDataset(BATCH_SIZE,TAKE_EVERY,TOKEN_SIZE))

    

    start_time = time.time()
    batch,indicies = next(ds)
    step = 1
    while(batch is not None):
        print_rank_0(f'time taken to generate batch: {time.time() - start_time:.3}s')
        res = score(neox_args,model,batch,TOKEN_SIZE)
        
        
        print_rank_0(f'{time.time() - start_time:3}s')
        start_time = time.time()
        batch,indicies = next(ds)
        step += 1
    records.close()
    
    


if __name__ == "__main__":
    main()
