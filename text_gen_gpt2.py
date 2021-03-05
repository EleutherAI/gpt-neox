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

"""Sample Generate GPT2"""

import os
import sys

from megatron.config_monster import ConfigMonster
from pretrain_gpt2 import model_provider

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import GPT2Model
from megatron.training import get_model
from megatron.text_generation_utils import generate_and_write_samples_unconditional
from megatron.text_generation_utils import generate_samples_input_from_file
from megatron.text_generation_utils import generate_samples_interactive


def main():
    """
    Provide load
    """

    _, conf, _, user_script_args = ConfigMonster().consume_args()

    initialize_megatron(args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}, args=user_script_args)

    # Set up model and load checkpoint.
    model = get_model(model_provider)
    args = get_args()
    if args.load is not None:
        print(f"Loading model: {args.load}")
        _ = load_checkpoint(model, None, None)
    else:
        print("WARNING: not loading a model")

    # Generate samples.
    if args.num_samples == 0:
        args.batch_size = 1
        if args.sample_input_file != None:
            generate_samples_input_from_file(model)
        else:
            generate_samples_interactive(model)
    else:
        generate_and_write_samples_unconditional(model)


if __name__ == "__main__":

    main()
