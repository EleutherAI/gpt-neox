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

"""Generate text / sample GPT2"""

import os
import sys
from pretrain_gpt2 import model_provider

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

from megatron import get_args, print_rank_0
from megatron.initialize import initialize_megatron
from megatron.training import setup_model_and_optimizer
from megatron.text_generation_utils import generate_and_write_samples_unconditional, generate_samples_input_from_file, \
    generate_samples_interactive
from megatron.utils import pipe_to_normal

from deepspeed import PipelineEngine

def main(extra_args_provider=None, get_key_value=True):
    """
    Generate text/sample model
    """

    initialize_megatron(args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}, extra_args_provider=extra_args_provider)

    args = get_args()

    if args.load is None:
        raise ValueError("`load` parameter must be supplied to load a model`")

    # Force checkpoint activations, don't load optimizer states
    args.checkpoint_activations = False
    args.partition_activations = False
    args.no_load_optim = True

    # Set up model and load checkpoint.
    model, _, _ = setup_model_and_optimizer(lambda: model_provider(use_wandb=False, inference=True, get_key_value=get_key_value))
    print_rank_0('Finished loading model')

    if args.text_gen_type == 'unconditional':
        print_rank_0('Generating samples unconditionally')
        assert args.genfile is not None
        generate_and_write_samples_unconditional(model)

    elif args.text_gen_type == 'input-file':
        print_rank_0(f'Generating {args.num_samples} samples from input file {args.sample_input_file}')
        assert args.sample_input_file is not None and args.sample_output_file is not None
        generate_samples_input_from_file(model)

    elif args.text_gen_type == 'interactive':
        print_rank_0(f'Generating {args.num_samples} samples interactively')
        raise NotImplementedError("Interactive generation is not implemented yet.")
        generate_samples_interactive(model)

    else:
        raise ValueError(f"`text-gen-type` either not specified or not recognised: {args.text_gen_type}")

if __name__ == "__main__":
    main()
