import os
import tarfile
import argparse
import deepspeed
import json
from collections import defaultdict


# helpers
def get_args():
    parser = argparse.ArgumentParser(description='GPTNeox Deepspeed Training Script')
    # Include DeepSpeed configuration arguments
    parser.add_argument('--model', type=str, default="gpt3_small")
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
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
