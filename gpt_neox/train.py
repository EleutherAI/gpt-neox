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

from tqdm.auto import tqdm, trange
from gpt_neox.arguments import get_argument_parser
from gpt_neox.utils import GPUMonitor
# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
GENERATE_LENGTH = 512
SEQ_LEN = 1024

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

def get_args():
    parser = get_argument_parser()
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    
    args = parser.parse_args()
    args.config_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'configs/base_deepspeed.json')
    args.deepspeed_config = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'configs/base_deepspeed.json')
    print(args)
    return args

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


def train_model():

    # instantiate GPT-like decoder model
    model = GPTNeoX(
        num_tokens = 256,
        dim = 512,
        seq_len = SEQ_LEN,
        depth = 6,
        heads = 8,
        dim_head = 64
    )
    model = AutoregressiveWrapper(model)
    #model.cuda()
    # prepare enwik8 data
    with gzip.open('./data/enwik8.gz') as file:
        X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
        trX, vaX = np.split(X, [int(90e6)])
        data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)


    train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
    val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)
    train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
    val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

    # optimizer

    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # training

    model_params = prepare_optimizer_parameters(model)

    train_args = get_args()
    print(train_args)

    # ds loader
    model_engine, optim, _, _ = deepspeed.initialize(args=train_args,
                                                        model=model,
                                                        optimizer=optim,
                                                        model_parameters=model_params)

    pbar = trange(NUM_BATCHES, mininterval=10., desc='Training Model', dynamic_ncols=True)
    monitor = GPUMonitor()
    for i in pbar:
        model.train()
        for __ in range(GRADIENT_ACCUMULATE_EVERY):
            loss = model_engine(next(train_loader))
            model_engine.backward(loss)

        model_engine.step()
        pbar.set_description(f'Training Loss: {loss.item()}')
        pbar.update()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()

        if i % VALIDATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                loss = model_engine(next(val_loader))
                pbar.write(f'Validation Loss: {loss.item()}')

        if i % GENERATE_EVERY == 0:
            model.eval()
            inp = random.choice(val_dataset)[:-1]
            prime = decode_tokens(inp)
            pbar.write(f'%s \n\n %s', (prime, '*' * 100))

            sample = model.generate(inp, GENERATE_LENGTH)
            output_str = decode_tokens(sample)
            pbar.write(output_str)
