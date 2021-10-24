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
from megatron.data.data_utils import build_train_valid_test_data_iterators
from megatron import mpu
from megatron.text_generation_utils import stream_tokens
from megatron.utils import print_rank_0, setup_for_inference_or_eval
import os
import time

class BatchedDataset(Thread):
    def __init__(self,batch_size,take_every,token_size,neox_args):
        super().__init__()
        self.batch_size = batch_size
        self.take_every = take_every
        self.token_size = token_size
        self.q = queue.Queue()
        self.ds, valid_ds, test_ds = build_train_valid_test_data_iterators(neox_args=neox_args)
    def run(self):
        
        tokens = []
        indicies = []
        val = 1
        idx = 0
        print_rank_0("Iterating through the dataset")
        for doc in self.ds:
            idx += 4 #Batch size of dataset is 4
            if(idx%self.take_every != 0):
                continue
            [tokens.append(i) for i in doc['text'][:self.token_size].numpy().tolist()]
            indicies.append(idx)
            if(val%self.batch_size == 0):
                self.q.put((tokens,indicies))
                
                while(self.q.qsize() > 10):
                    time.sleep(50)
                indicies = []
                tokens = []
            val += 1
        self.q.put((None,None))
        self.q.task_done()



def score(neox_args,model,data,token_size=64):
    '''Calculates the memorization metric for the given input tokens
    '''
    
    inp = [i[:token_size//2] for i in data]
    res = stream_tokens(
        neox_args=neox_args, 
        model=model,
        context_tokens = inp,
        maximum_tokens = token_size, 
        recompute = neox_args.recompute, 
        temperature = neox_args.temperature,
        top_k = neox_args.top_k, 
        top_p = neox_args.top_p
    )
    res = res[:,token_size//2:token_size,:].cpu().transpose(0,1)
    ground_truth = [i[token_size//2:token_size] for i in data]
    return memorization_metric(res,torch.tensor(ground_truth))



def main():
    BATCH_SIZE = 128
    RESULTS_PATH = '/home/mchorse/gpt-neox/memorization_results_neox_dense_small_v2.tfrecords'
    TOKEN_SIZE = 64
    TAKE_EVERY = 32

    records = TFrecordCreator(RESULTS_PATH) #store results
    
    model, neox_args = setup_for_inference_or_eval()    

    ds = BatchedDataset(BATCH_SIZE,TAKE_EVERY,TOKEN_SIZE,neox_args)
    ds.start()
    

    start_time = time.time()
    batch,indicies = ds.q.get()
    step = 1
    while(batch is not None):
        print_rank_0(f'time taken to generate batch: {time.time() - start_time:.3}s')
        start_time = time.time()

        res = score(neox_args,model,batch,TOKEN_SIZE)
        for i,j in zip(res,indicies):
            records.write(i,j)
                
        print_rank_0(f'Current model generation time: {time.time() - start_time:.3}s for index {indicies[-1]}')
        start_time = time.time()
        batch,indicies = ds.q.get(10)
        step += 1
    records.close()
    
    


if __name__ == "__main__":
    main()
