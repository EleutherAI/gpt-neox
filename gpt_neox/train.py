from gpt_neox import GPTNeoX, AutoregressiveWrapper

import os
import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
import deepspeed
import json
from tqdm.auto import tqdm, trange
from gpt_neox.utils import GPUMonitor, DictArgs

# constants

BaseConfig = {
    'model': {
        'num_batches': int(1e5),
        'batch_size': 8,
        'gradient_accumulate_every': 4,
        'learning_rate': 1e-4,
        'validate_every': 100,
        'generate_every': 500,
        'generate_length': 512,
        'seq_len': 1024,
        'num_tokens': 256,
        'dim': 512,
        'depth': 6,
        'heads': 8,
        'dim_head': 64,
    },
    'ds': {
        'local_rank': -1,
        'output_dir': './model',
        'deepspeed_config': './configs/base_deepspeed.json',
        'deepspeed': True,
        'save_interval': 100,
    }
}


# helpers

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


class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len

def load_wiki_dataset():
    with gzip.open('./data/enwik8.gz') as file:
        X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
        trX, vaX = np.split(X, [int(90e6)])
        data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)
        return data_train, data_val


def update_ds_config(model_config, deepspeed_config):
    dsc = json.load(open(deepspeed_config['deepspeed_config'], 'r'))
    dsc['train_batch_size'] = model_config['batch_size']
    dsc['gradient_accumulation_steps'] = model_config['gradient_accumulate_every']
    dsc['optimizer']['params']['lr'] = model_config['learning_rate']
    json.dump(dsc, open(deepspeed_config['deepspeed_config'], 'w'))


def train_model(model_config=None, deepspeed_config=None):
    config = BaseConfig
    _mc = config['model']
    if model_config:
        _mc.update(model_config)
    _dsc = config['ds']
    if deepspeed_config:
        _dsc.update(deepspeed_config)
    update_ds_config(_mc, _dsc)
    os.makedirs(_dsc['output_dir'], exist_ok=True)
    # instantiate GPT-like decoder model
    model = GPTNeoX(
        num_tokens = _mc['num_tokens'],
        dim = _mc['dim'],
        seq_len = _mc['seq_len'],
        depth = _mc['depth'],
        heads = _mc['heads'],
        dim_head = _mc['dim_head']
    )
    model = AutoregressiveWrapper(model)

    data_train, data_val = load_wiki_dataset()

    train_dataset = TextSamplerDataset(data_train, _mc['seq_len'])
    val_dataset   = TextSamplerDataset(data_val, _mc['seq_len'])
    train_loader  = cycle(DataLoader(train_dataset, batch_size = _mc['batch_size']))
    val_loader    = cycle(DataLoader(val_dataset, batch_size = _mc['batch_size']))

    # optimizer
    optim = torch.optim.Adam(model.parameters(), lr=_mc['learning_rate'])

    # training
    model_params = prepare_optimizer_parameters(model)
    train_args = DictArgs(_dsc)

    # ds loader
    model_engine, optim, _, _ = deepspeed.initialize(args=train_args,
                                                        model=model,
                                                        optimizer=optim,
                                                        model_parameters=model_params)

    pbar = trange(_mc['num_batches'], mininterval=10., desc='Training Model', dynamic_ncols=True)
    monitor = GPUMonitor()
    for step in pbar:
        model.train()
        for __ in range(_mc['gradient_accumulate_every']):
            loss = model_engine(next(train_loader))
            model_engine.backward(loss)

        model_engine.step()
        pbar.set_description(f'Training Loss: {loss.item()}')
        pbar.update()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()

        if step % _mc['validate_every'] == 0:
            model.eval()
            with torch.no_grad():
                loss = model_engine(next(val_loader))
                pbar.write(f'Validation Loss: {loss.item()}')

        if step % _mc['generate_every'] == 0:
            model.eval()
            inp = random.choice(val_dataset)[:-1]
            prime = decode_tokens(inp)
            pbar.write(f'{"----" * 50}')
            pbar.write(f"Prime Inputs: {prime}")
            pbar.write(f'{"----" * 50}')

            sample = model.generate(inp, _mc['generate_length'])
            output_str = decode_tokens(sample)
            pbar.write(f"Decoded Outputs: {output_str}")
            pbar.write(f'{"----" * 50}')
        
        if (step+1) % train_args.save_interval == 0:
            pbar.write(f'Saving Checkpoint at {step+1} to {train_args.output_dir}')
            model_engine.save_checkpoint(train_args.output_dir)


if __name__ == '__main__':
    train_model()