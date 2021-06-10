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

from megatron.neox_arguments import NeoXArgs
from megatron import initialize_megatron
from megatron.training import setup_model_and_optimizer
from megatron.utils import print_rank_0

from megatron.text_generation_utils import generate_samples_input_from_file, generate_samples_from_prompt, generate_samples_unconditional, generate_samples_interactive

if __name__ == "__main__":
    """
    Generate text/sample model
    """

    neox_args = NeoXArgs.consume_neox_args(overwrite_values={
        "checkpoint_activations": False,
        "partition_activations": False,
        "no_load_optim": True,
    })
    neox_args.configure_distributed_args()
    neox_args.build_tokenizer()
    
    if neox_args.load is None:
        raise ValueError("`load` parameter must be supplied to load a model`")

    # initialize megatron
    initialize_megatron(neox_args)

    # set up model and load checkpoint.
    model, _, _ = setup_model_and_optimizer(neox_args=neox_args, inference=True, get_key_value=True) # we use setup_model_and_optimizer instead of get_model in order to initialize deepspeed
    print_rank_0('Finished loading model')

    if neox_args.text_gen_type == 'unconditional':
        print_rank_0('Generating samples unconditionally')
        assert neox_args.sample_output_file is not None
        generate_samples_unconditional(
            neox_args=neox_args, 
            model=model,
            number_of_samples=neox_args.num_samples,
            output_file=neox_args.sample_output_file,
            maximum_tokens = neox_args.maximum_tokens, 
            recompute = neox_args.recompute, 
            temperature = neox_args.temperature,
            top_k = neox_args.top_k, 
            top_p = neox_args.top_p
        )

    elif neox_args.text_gen_type == 'input-file':
        print_rank_0(f'Generating samples from input file {neox_args.sample_input_file}')
        assert neox_args.sample_input_file is not None
        generate_samples_input_from_file(
            neox_args=neox_args, 
            model=model,
            input_file=neox_args.sample_input_file,
            output_file=neox_args.sample_output_file,
            maximum_tokens = neox_args.maximum_tokens, 
            recompute = neox_args.recompute, 
            temperature = neox_args.temperature,
            top_k = neox_args.top_k, 
            top_p = neox_args.top_p
        )

    elif neox_args.text_gen_type == 'interactive':
        generate_samples_interactive(
            neox_args=neox_args, 
            model=model,
            recompute = neox_args.recompute, 
            temperature = neox_args.temperature,
            top_k = neox_args.top_k, 
            top_p = neox_args.top_p
        )

    else:
        raise ValueError(f"`text-gen-type` either not specified or not recognised: {neox_args.text_gen_type}")

