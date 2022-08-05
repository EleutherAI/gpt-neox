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

"""Utils for MTF datasets."""

import time

import numpy as np
import torch

from megatron import print_rank_0
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset

from megatron.data.gpt2_dataset import _build_shuffle_idx

# TODO(Hailey): this is a temp file containing new data util/helper fns for MTF. 
# Should be added into data_utils.py when I'm done?



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
