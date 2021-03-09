#!/usr/bin/env python
# coding=utf-8
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

"""Generate text / sample GPT2"""
import json
import os
import sys
from json import JSONDecodeError

import deepspeed

from megatron.config_monster import ConfigMonster
from pretrain_gpt2 import model_provider, get_batch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import GPT2Model, GPT2ModelPipe
from megatron.training import get_model, setup_model_and_optimizer
from megatron.text_generation_utils import generate_and_write_samples_unconditional, generate_samples_input_from_file_2
from megatron.text_generation_utils import generate_samples_input_from_file
from megatron.text_generation_utils import generate_samples_interactive


def main():
    """
    Generate text/sample model
    """

    initialize_megatron(args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})

    args = get_args()

    if args.load is None:
        raise ValueError("`load` parameter must be supplied to load a model`")

    # Force disable PP, checkpoint activations and always weight tie
    args.pipe_parallel_size = 0
    args.checkpoint_activations = False
    args.partition_activations = False
    args.no_weight_tying = False

    # Set up model and load checkpoint.
    model, optimizer, lr_scheduler = setup_model_and_optimizer(lambda: model_provider(use_wandb=False))

    print('Finished loading model')

    if args.text_gen_type == 'unconditional':
        print('Generating samples unconditionally')
        assert args.genfile is not None
        generate_and_write_samples_unconditional(model)

    elif args.text_gen_type == 'input-file':
        print(f'Generating {args.num_samples} samples from input file {args.sample_input_file}')
        assert args.sample_input_file is not None and args.sample_output_file is not None
        generate_samples_input_from_file_2(model)

    elif args.text_gen_type == 'interactive':
        print(f'Generating {args.num_samples} samples interactively')
        raise NotImplementedError("Interactive generation is not implemented yet.")
        generate_samples_interactive(model)

    else:
        raise ValueError(f"`text-gen-type` either not specified or not recognised: {args.text_gen_type}")

if __name__ == "__main__":
    main()
