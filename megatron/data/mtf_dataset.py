# coding=utf-8
# Copyright (c) 2022, EleutherAI contributors
# This file is based on code by the authors denoted below and has been modified from its original version. 
# TODO: add attribution to Bigscience Meg-DS fork + authors?
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

"""Multitask Finetune style dataset."""

import time

import numpy as np
import torch

from megatron import print_rank_0
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset

class MTFDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        name,
        data_prefix,
        data_impl,
        skip_warmup,
        documents,
    ):
        # Params to store.
        self.name = name

        # Dataset.
        self.input_indexed_dataset = get_indexed_dataset(data_prefix, is_input=True, data_impl=data_impl, skip_warmup=skip_warmup)
        self.target_indexed_dataset = get_indexed_dataset(data_prefix, is_input=False, data_impl=data_impl, skip_warmup=skip_warmup)

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < self.input_indexed_dataset.sizes.shape[0]
        assert np.max(documents) < self.target_indexed_dataset.sizes.shape[0]
        assert self.input_indexed_dataset.sizes.shape[0] == self.target_indexed_dataset.sizes.shape[0]

    def __len__(self):
        return len(self.input_indexed_dataset)

    def __getitem__(self, idx):
        input_tokens = self.input_indexed_dataset.get(idx)
        target_tokens = self.target_indexed_dataset.get(idx)

        assert len(input_tokens) > 0
        assert len(target_tokens) > 0

        return {
            'input_tokens': input_tokens,
            'target_tokens': target_tokens,
        }

    def size(self, index):
        return {
            'input_tokens': self.input_indexed_dataset.size(index),
            'target_tokens': self.target_indexed_dataset.size(index),
        }

def get_indexed_dataset(data_prefix: str, is_input: bool, data_impl: str, skip_warmup: bool):
    if is_input:
        field = "inputs"
    else:
        field = "targets"

    return get_indexed_dataset_(f"{data_prefix}_{field}_document", data_impl, skip_warmup)

def get_indexed_dataset_(path, data_impl, skip_warmup):
    """Build indexed dataset."""
    print_rank_0(' > building dataset index ...')
    start_time = time.time()
    indexed_dataset = make_indexed_dataset(path,
                                           data_impl,
                                           skip_warmup)
    print_rank_0(' > finished creating indexed dataset in {:4f} '
                 'seconds'.format(time.time() - start_time))
    print_rank_0('    number of documents: {}'.format(
        indexed_dataset.sizes.shape[0]))

    return indexed_dataset