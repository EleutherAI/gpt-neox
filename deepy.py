#!/usr/bin/env python

"""
Helper script to enable wandb with deepspeed. Drop in replacement for `deepspeed` command.

i.e. `$ deepy.py --hosts ....` == `$ deepspeed --hosts ....`
"""

import shortuuid
import requests
import sys
import os
import deepspeed
from deepspeed.launcher.runner import main

# Generate unique run group name
wandb_group = shortuuid.uuid()
sys.argv.extend(['--group_name', wandb_group])

# Extract wandb API key and inject into worker environments
wandb_token = requests.utils.get_netrc_auth('https://api.wandb.ai')
if wandb_token is not None:
    wandb_token = wandb_token[1]
    deepspeed.launcher.runner.EXPORT_ENVS.append('WANDB_API_KEY')
    os.environ['WANDB_API_KEY'] = wandb_token

if __name__ == '__main__':
    main()

