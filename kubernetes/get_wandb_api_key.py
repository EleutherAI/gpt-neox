#!/usr/bin/env python

"""
Get Weights and Biases API key
"""

import requests
import os

def get_wandb_api_key():
    if 'WANDB_API_KEY' in os.environ:
        return os.environ['WANDB_API_KEY']

    wandb_token = requests.utils.get_netrc_auth('https://api.wandb.ai')

    if wandb_token is not None:
        return wandb_token[1]

if __name__ == "__main__":
    api_key = get_wandb_api_key()
    if api_key is not None:
        print(api_key)
