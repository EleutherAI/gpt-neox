# coding=utf-8
# Copyright (c) 2021, EleutherAI contributors
# This file is based on code by the authors denoted below and has been modified from its original version.
#
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

"""Main tasks functionality."""

import os
import sys
from functools import partial
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

from megatron import print_rank_0
from megatron.training import forward_step
from megatron.initialize import initialize_megatron
from megatron.neox_arguments import NeoXArgs
from megatron import initialize_megatron
from megatron.training import setup_model_and_optimizer
from megatron.utils import print_rank_0, is_mp_rank_0, is_local_main, setup_for_inference_or_eval
from megatron.utils import ddb

from tqdm import tqdm 

import torch
import lm_eval
from lm_eval.base import CacheHook

from lm_eval.models.gpt2 import GPT2LM
from lm_eval import tasks, evaluator
from lm_eval import utils
import torch.nn.functional as F

import collections
import itertools
import random
import lm_eval.metrics

from adaptor import run_eval_harness


if __name__ == "__main__":
    model, neox_args = setup_for_inference_or_eval(inference=False, get_key_value=False)
    run_eval_harness(model, forward_step, neox_args)
