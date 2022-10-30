#!/usr/bin/env python
# coding=utf-8
from megatron import text_generation_utils
from megatron import utils as megatron_utils
from megatron.data import data_utils
from megatron.neox_arguments import NeoXArgs
from tqdm import trange
import numpy as np
models = {
    '13B': [143000],
    '13B_deduped': [143000],
    '6.7B': [143000],
    '6.7B_deduped': [143000],
    '2.7B': [143000],
    '1.3B': [71500],
    '1.3B_deduped': [71500],
    '800M': [143000],
    '800M_deduped': [143000],
    '350M': [71500],
    '350M_deduped': [71500],
    '125M': [71500],
    '125M_deduped': [71500]
}
filepath = '/fsx/orz/transformer-memorization'

import os
import pandas as pd
from tqdm.auto import tqdm
from megatron import mpu

tqdm.pandas()

memorization_results = {}
for model, checkpoints in models.items():
    for checkpoint in tqdm(checkpoints, desc=model):
        filename = os.path.join(filepath, f'memorization_results_{model}-{checkpoint}.csv.hdf')
        model_name = f'{model}-{checkpoint}'
        try:
            memorization_results[model_name] = pd.read_hdf(filename, key='memorization')
        except FileNotFoundError:
            csv = pd.read_csv( os.path.join(filepath, f'memorization_results_{model}-{checkpoint}.csv'))
            csv.to_hdf(filename, key='memorization', index=False)
            memorization_results[model_name] = csv

# Manually initializing cpu backend

import torch.distributed as dist
import os
neox_args = NeoXArgs.consume_neox_args()
dist.init_process_group(
    "gloo", 
    rank = int(os.environ['RANK']),
    world_size= int(os.environ['WORLD_SIZE']),
)

from megatron.initialize import initialize_megatron

neox_args.configure_distributed_args()
neox_args.build_tokenizer()
initialize_megatron(neox_args, allow_no_cuda=True)
ds,_,_ = data_utils.build_train_valid_test_data_iterators(neox_args=neox_args)
texts = []
tot_len = len(memorization_results['13B-143000']['accuracy'])
idx = 0
for i in trange(tot_len//neox_args.train_batch_size + 1):
    batch = next(ds)['text']
    curr_idx = mpu.get_data_parallel_rank()*neox_args.train_micro_batch_size_per_gpu
    for j in range(neox_args.train_micro_batch_size_per_gpu):
        acc_one_models = []
        for model in models:
            model_name = f'{model}-{models[model][0]}'
            evals = memorization_results[model_name]
            if(evals['accuracy'].iloc[curr_idx] == 1):
                acc_one_models.append(model)
        if(acc_one_models):
            texts.append((curr_idx,acc_one_models, neox_args.tokenizer.detokenize(batch[j,:64].cpu().numpy())))
        curr_idx += 1
    idx += neox_args.train_batch_size


import json
with open(f'texts_{mpu.get_data_parallel_rank()}.json', 'w') as f:
    json.dump(texts, f)

