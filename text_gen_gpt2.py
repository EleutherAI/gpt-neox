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


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from megatron.neox_arguments import NeoXArgs
from megatron import print_rank_0
from megatron.initialize import initialize_megatron
from megatron.training import setup_model_and_optimizer
from megatron.text_generation_utils import generate_and_write_samples_unconditional, generate_samples_input_from_file, generate_samples_interactive


def main():
    """
    Generate text/sample model
    """

    neox_args = NeoXArgs.consume_neox_args()
    neox_args.build_tokenizer()
    
    # Force checkpoint activations, don't load optimizer states
    # neox_args.update_value is used in order to assert that the attributes to be set are existing (no typos here)
    neox_args.update_value("checkpoint_activations", False)
    neox_args.update_value("partition_activations", False)
    neox_args.update_value("no_load_optim", True)

    if neox_args.load is None:
        raise ValueError("`load` parameter must be supplied to load a model`")

    # initialize megatron
    initialize_megatron(neox_args)

    # set up model and load checkpoint.
    model, _, _ = setup_model_and_optimizer(neox_args=neox_args, inference=True, get_key_value=True) # we use setup_model_and_optimizer instead of get_model in order to initialize deepspeed
    print_rank_0('Finished loading model')

    if neox_args.text_gen_type == 'unconditional':
        print_rank_0('Generating samples unconditionally')
        assert neox_args.genfile is not None
        generate_and_write_samples_unconditional(neox_args=neox_args, model=model)

    elif neox_args.text_gen_type == 'input-file':
        print_rank_0(f'Generating {args.num_samples} samples from input file {args.sample_input_file}')
        assert neox_args.sample_input_file is not None and args.sample_output_file is not None
        generate_samples_input_from_file(neox_args=neox_args, model=model)

    elif neox_args.text_gen_type == 'interactive':
        print_rank_0(f'Generating {neox_args.num_samples} samples interactively')
        raise NotImplementedError("Interactive generation is not implemented yet.")
        generate_samples_interactive(neox_args=neox_args, model=model)

    else:
        raise ValueError(f"`text-gen-type` either not specified or not recognised: {neox_args.text_gen_type}")

if __name__ == "__main__":
    main()
