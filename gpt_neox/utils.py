import os
import tarfile
import argparse
import deepspeed
import json
from collections import defaultdict
import shutil
import re
import random
import numpy as np
import requests
import torch


# helpers
def get_args():
    parser = argparse.ArgumentParser(description='GPTNeox Deepspeed Training Script')
    # Include DeepSpeed configuration arguments
    parser.add_argument('--model', type=str, default="gpt3_small")
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--group_name', type=str, default=None, help='Group name used by wandb')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def get_params(model):
    model_path = model if model.endswith(".json") else f"./configs/{model}.json"
    with open(model_path) as f:
        params = json.load(f)
    return defaultdict(lambda: None, params)


def is_main(args):
    """
    returns True if process is being run on the main GPU
    """
    return args.local_rank in [0, -1]


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def save_ds_checkpoint(iteration, model, params, keep_n_latest_checkpoints=None, is_main=None):
    """Save a model checkpoint."""
    iteration = str(iteration)
    sd = {}
    sd['iteration'] = iteration
    if keep_n_latest_checkpoints is not None:
        assert is_main is not None
    # rng states.
    if params.get('save_rng', True):
        sd['random_rng_state'] = random.getstate()
        sd['np_rng_state'] = np.random.get_state()
        sd['torch_rng_state'] = torch.get_rng_state()
        sd['cuda_rng_state'] = torch.cuda.get_rng_state()
        sd['rng_tracker_states'] = model.mpu.get_cuda_rng_tracker().get_states()

    checkpoint_dir = params.get('checkpoint_dir', None)
    assert checkpoint_dir is not None, 'add "checkpoint_dir" to your model params to enable checkpoint saving'
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    if keep_n_latest_checkpoints is not None:
        all_checkpoints = os.listdir(checkpoint_dir)
        checkpoint_dirs = natural_sort(all_checkpoints)
        checkpoint_dirs = [item for item in checkpoint_dirs if os.path.isdir(os.path.join(checkpoint_dir, item))]
        checkpoint_dirs = [str(i) for i in checkpoint_dirs]
        n = len(checkpoint_dirs) - keep_n_latest_checkpoints
        n = 0 if n < 0 else n
        to_delete = checkpoint_dirs[:n+1]
        if to_delete:
            if is_main:
                print(f'WARNING: deleting checkpoint dirs {to_delete} in {checkpoint_dir}')
                [shutil.rmtree(os.path.join(checkpoint_dir, item)) for item in to_delete]
    model.save_checkpoint(checkpoint_dir, iteration, client_state=sd)


def load_ds_checkpoint(model, params, iteration=None):
    """Load a model checkpoint."""
    if iteration is not None:
        iteration = str(iteration)
        
    checkpoint_dir = params.get('checkpoint_dir', None)
    assert checkpoint_dir is not None, 'add "checkpoint_dir" to your model params to enable checkpoint loading'
    print(f'Loading latest checkpoint from {checkpoint_dir}')

    checkpoint_name, sd = model.load_checkpoint(checkpoint_dir, iteration)
    if checkpoint_name is None:
        print("Unable to load checkpoint.")
        return iteration if iteration is not None else 0

    # rng states.
    if params.get('load_rng', True):
        try:
            random.setstate(sd['random_rng_state'])
            np.random.set_state(sd['np_rng_state'])
            torch.set_rng_state(sd['torch_rng_state'])
            torch.cuda.set_rng_state(sd['cuda_rng_state'])
            model.mpu.get_cuda_rng_tracker().set_states(sd['rng_tracker_states'])
        except KeyError:
            print(f'Unable to load rngs from checkpoint {checkpoint_name}, exiting. ')
            exit()
    torch.distributed.barrier()
    print(f'successfully loaded {checkpoint_name}')
    iteration = int(os.path.basename(os.path.dirname(checkpoint_name)))
    return iteration


def get_all_files(filetype, files_dir):
    files = []
    for (dir_path, _, filenames) in os.walk(files_dir):
        for filename in filenames:
            if filename.endswith(".{}".format(filetype)):
                file_path = os.path.join(dir_path, filename)
                files.append(file_path)
    return files


def extract_tarfile(tarfile_path, extract_dir=None):
    dataset_tar = tarfile.open(tarfile_path, "r:gz")
    os.makedirs(extract_dir, exist_ok=True)
    dataset_tar.extractall(extract_dir)
    dataset_tar.close()


def cycle(loader):
    while True:
        for data in loader:
            yield data


def decode_token(token):
    return str(chr(max(32, token)))


def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))


def prepare_optimizer_parameters(model):
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    weight_decay = 0.01
    optimizer_grouped_parameters = [{
        'params':
            [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
            weight_decay
    }, {
        'params':
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
            0.0
    }]
    return optimizer_grouped_parameters


class DictArgs(dict):
    def __init__(self, config):
        for k, v in config.items():
            self[k] = v

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


def get_wandb_api_key():
    """ Get Weights and Biases API key from ENV or .netrc file. Otherwise return None """
    if 'WANDB_API_KEY' in os.environ:
        return os.environ['WANDB_API_KEY']

    wandb_token = requests.utils.get_netrc_auth('https://api.wandb.ai')

    if wandb_token is not None:
        return wandb_token[1]