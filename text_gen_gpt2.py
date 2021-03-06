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
from pretrain_gpt2 import model_provider

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import GPT2Model, GPT2ModelPipe
from megatron.training import get_model, setup_model_and_optimizer
from megatron.text_generation_utils import generate_and_write_samples_unconditional
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

    # Set up model and load checkpoint.
    model, optimizer, lr_scheduler = setup_model_and_optimizer(lambda: model_provider(use_wandb=False))

    print('Finished loading model')

    print('Generating samples unconditionally')
    generate_and_write_samples_unconditional(model)

    exit()

    if args.text_gen_type == 'unconditional':
        pass
    else:
        raise ValueError(f"`text-gen-type` either not specified or not recognised: {args.text_gen_type}")

    # Generate samples.
    if args.num_samples == 0:
        args.batch_size = 1
        if args.sample_input_file is not None:
            print(f'Generating {args.num_samples} samples from input file {args.sample_input_file}')
            generate_samples_input_from_file(model)
        else:
            print(f'Generating {args.num_samples} samples interactively')
            generate_samples_interactive(model)
    else:
        print('Generating samples unconditionally')
        generate_and_write_samples_unconditional(model)


if __name__ == "__main__":
    main()
