#!/usr/bin/env python
# # Copyright (c) 2025, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from megatron.utils import setup_for_inference_or_eval
from megatron.text_generation_utils import generate_from_config


def main(input_args=None, overwrite_values=None):
    """
    Generate text/sample model
    """
    model, neox_args = setup_for_inference_or_eval(
        use_cache=True, input_args=input_args, overwrite_values=overwrite_values
    )
    if neox_args.recompute:
        model.module.inference_mode(
            use_cache=False
        )  # don't use kv cache if recomputing
    generate_from_config(neox_args, model)


if __name__ == "__main__":
    main()
