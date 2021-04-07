#!/usr/bin/env python
# Copyright (c) 2021, EleutherAI contributors
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

from socket import gethostname

import shortuuid
import os
import deepspeed
from deepspeed.launcher.runner import main
import requests
import subprocess

from megatron.config_monster import ConfigMonster
import logging

from megatron.logging import Tee

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def get_wandb_api_key():
    """ Get Weights and Biases API key from ENV or .netrc file. Otherwise return None """
    if 'WANDB_API_KEY' in os.environ:
        return os.environ['WANDB_API_KEY']

    wandb_token = requests.utils.get_netrc_auth('https://api.wandb.ai')

    if wandb_token is not None:
        return wandb_token[1]


def get_git_commit_hash():
    """ Gets the git commit hash of your current repo (if it exists) """
    try:
        git_hash = git_hash = subprocess.check_output(["git", "describe", "--always"]).strip()
        git_hash = git_hash.decode()
    except subprocess.CalledProcessError:
        git_hash = None
    return git_hash


# Generate unique run group name
wandb_group = shortuuid.uuid()
extra_conf = {
    'wandb_group': wandb_group,
    'git_hash': get_git_commit_hash()
}

# Extract wandb API key and inject into worker environments
wandb_token = get_wandb_api_key()
if wandb_token is not None:
    deepspeed.launcher.runner.EXPORT_ENVS.append('WANDB_API_KEY')
    os.environ['WANDB_API_KEY'] = wandb_token

old_style_args, conf = ConfigMonster().consume_args(extra_conf=extra_conf)

if 'log-dir' in conf:
    os.makedirs(conf['log-dir'], exist_ok=True)
    file_prefix = os.path.join(conf['log-dir'], '0-deepy')
    Tee(file_prefix + '_stdout.txt', err=False)
    Tee(file_prefix + '_stderr.txt', err=True)

if __name__ == '__main__':
    main(old_style_args)
