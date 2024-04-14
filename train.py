# Copyright (c) 2024, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from megatron.neox_arguments import NeoXArgs
from megatron.training import pretrain

import os
import numpy as np


def main(input_args=None, overwrite_values=None):
    neox_args = NeoXArgs.consume_neox_args(
        input_args=input_args, overwrite_values=overwrite_values
    )
    neox_args.configure_distributed_args()
    neox_args.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab

    if neox_args.load.split('/')[-1].startswith('JOB'):
        if 'scratch' in neox_args.load:
            training_mode = 'scratch'
        elif 'finetune' in neox_args.load:
            training_mode = 'finetune'
        else:
            training_mode = 'resume'

    elif neox_args.load == 'none':
        training_mode = 'scratch'
    elif neox_args.finetune:
        training_mode = 'finetune'
    else:
        training_mode = 'resume'

    dir_str = "JOB-{}_{}_it-{}_wu-{}_mxlr-{}_mnlr-{}_sch-{}_tr-{}_{}".format(
        "ENTER_YOUR_JOBID_IN_TRAIN.PY",# os.environ['LSB_JOBID'],
        neox_args.identifier_string.replace('_',"-"),
        neox_args.train_iters,
        neox_args.warmup,
        neox_args.optimizer['params']['lr'], 
        neox_args.min_lr,
        neox_args.lr_decay_style,
        neox_args.train_dataset_name.replace('_',"-"),
        training_mode)


    
    neox_args.tensorboard_dir = os.path.join(neox_args.tensorboard_dir, dir_str)
    neox_args.save = os.path.join(neox_args.save, dir_str)
    print("NEOX ARGS tensorboard_dir: ", neox_args.tensorboard_dir)
    print("NEOX ARGS save: ", neox_args.save)
    # exit(0)



    neox_args.initialize_tensorboard_writer()  # is initialized if tensorboard directory is defined
    pretrain(neox_args=neox_args)


if __name__ == "__main__":
    main()
