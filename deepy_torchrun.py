#!/usr/bin/env python
# Copyright (c) 2025, EleutherAI
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

"""Launch GPT-NeoX with torchrun while keeping the deepy.py CLI."""

import logging
import os
import subprocess
import sys


def main(input_args=None):
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

    from megatron.neox_arguments import NeoXArgs
    from megatron.utils import get_wandb_api_key

    neox_args = NeoXArgs.consume_deepy_args(input_args)
    deepspeed_main_args = neox_args.get_deepspeed_main_args()

    # Extract wandb API key and inject into worker environments
    wandb_token = get_wandb_api_key(neox_args=neox_args)
    if wandb_token is not None:
        os.environ["WANDB_API_KEY"] = wandb_token

    slurm_env = {
        "nnodes": os.environ.get("SLURM_JOB_NUM_NODES"),
        "nproc_per_node": os.environ.get("SLURM_GPUS_ON_NODE"),
        "master_addr": os.environ.get("MASTER_ADDR"),
        "master_port": os.environ.get("MASTER_PORT"),
        "node_rank": os.environ.get("RANK")
    }
    missing = [name for name, value in slurm_env.items() if not value]
    if missing:
        raise RuntimeError(
            "deepy_torchrun.py expects a Slurm-style launch environment and is "
            f"missing: {', '.join(missing)}"
        )

    # DeepSpeed launcher args come first; torchrun only needs the target script and
    # the arguments that would normally be forwarded to it.
    user_script_idx = deepspeed_main_args.index(neox_args.user_script)
    cmd = [
        "torchrun",
        "--nnodes",
        slurm_env["nnodes"],
        "--nproc-per-node",
        slurm_env["nproc_per_node"],
        "--master-addr",
        slurm_env["master_addr"],
        "--master-port",
        slurm_env["master_port"],
        "--node-rank",
        slurm_env["node_rank"],
        "--log-dir",
        os.path.join(os.getcwd(), "logs"),
        *deepspeed_main_args[user_script_idx:],
    ]

    env = os.environ.copy()
    curr_path = os.path.abspath(".")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = curr_path + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = curr_path

    logging.info("Running command: %s", " ".join(cmd))
    result = subprocess.run(cmd, env=env, check=False)

    # In case of failure must propagate the error-condition back to the caller (usually shell). The
    # actual error and traceback should have been printed in the subprocess, so in order to avoid
    # unnecessary noise we just quietly exit here with the same code as the subprocess
    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()