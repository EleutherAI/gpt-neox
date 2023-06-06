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
from megatron.neox_arguments import NeoXArgs
from megatron.training import pretrain
from megatron.neox_arguments.deepspeed_args import NeoXArgsDeepspeedRunner

import argparse
import os
os.environ['NCCL_DEBUG'] = 'VERSION'
#os.environ['SMDATAPARALLEL_LMC_ENABLE'] = '1'
#os.environ['SMDDP_AG_SCRATCH_BUFFER_SIZE_BYTES'] = '8192'
#os.environ['SMDDP_AG_SORT_BUFFER_SIZE_BYTES'] = '8192'
if os.environ["OMPI_COMM_WORLD_RANK"] == "0":
    pass
    #os.environ["NCCL_DEBUG"] = "INFO"
    #os.environ["NCCL_DEBUG_SUBSYS"] = "COLL"
import smdistributed.dataparallel.torch.torch_smddp


def parse_config():
    parser = argparse.ArgumentParser(
        description="GPT-NeoX Configuration", allow_abbrev=False
    )
    group = parser.add_argument_group(title="Training Configuration")

    group.add_argument(
        "--conf",
        type=str,
        nargs="+",
        help="Configuration file path. Multiple files can be provided and will be merged.",
    )
    args_parsed = parser.parse_args()
    conf_files = args_parsed.conf
    neox_args = NeoXArgs.from_ymls(
        paths_to_yml_files=conf_files, overwrite_values=dict()
    )
    neox_args.print()
    return neox_args


if __name__ == "__main__":
    neox_args = parse_config()
    print(neox_args)
    neox_args.configure_distributed_args()
    neox_args.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab
    neox_args.initialize_tensorboard_writer()  # is initialized if tensorboard directory is defined
    pretrain(neox_args=neox_args)

