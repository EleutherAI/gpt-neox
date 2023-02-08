# Copyright (c) 2021, EleutherAI
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

"""Train"""
import os
import logging 
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

from megatron.neox_arguments import NeoXArgs
from megatron.training import pretrain



if __name__ == "__main__":
    # 复用deepy.py的代码，在这里生成所有的参数，前三个参数是不需要的，直接丢掉
    neox_args = NeoXArgs.consume_deepy_args()
    deepspeed_main_args = neox_args.get_deepspeed_main_args()
    args=deepspeed_main_args[2:]
    
    # 传入我们生成的参数列表
    neox_args = NeoXArgs.consume_neox_args(args=args)
    neox_args.configure_distributed_args()
    neox_args.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab
    neox_args.initialize_tensorboard_writer()  # is initialized if tensorboard directory is defined
    pretrain(neox_args=neox_args)