#!/usr/bin/env python

"""
Helper script to enable wandb with deepspeed. Drop in replacement for `deepspeed` command.

i.e. `$ deepy.py --hosts ....` == `$ deepspeed --hosts ....`
"""

import shortuuid
import sys
import os
import deepspeed
from deepspeed.launcher.runner import main
import requests

from megatron.config_monster import ConfigMonster


def get_wandb_api_key():
    """ Get Weights and Biases API key from ENV or .netrc file. Otherwise return None """
    if 'WANDB_API_KEY' in os.environ:
        return os.environ['WANDB_API_KEY']

    wandb_token = requests.utils.get_netrc_auth('https://api.wandb.ai')

    if wandb_token is not None:
        return wandb_token[1]

# Generate unique run group name
wandb_group = shortuuid.uuid()
extra_conf = {
    'wandb_group': wandb_group
}

wandb_team = os.environ.get('WANDB_TEAM')
if wandb_team:
    extra_conf['wandb_team'] = wandb_team

# Extract wandb API key and inject into worker environments
wandb_token = get_wandb_api_key()
if wandb_token is not None:
    deepspeed.launcher.runner.EXPORT_ENVS.append('WANDB_API_KEY')
    os.environ['WANDB_API_KEY'] = wandb_token

old_style_args = ConfigMonster().consume_args(extra_conf=extra_conf)

### LOG CONFIG TO WANDB

if __name__ == '__main__':
    main(old_style_args)

