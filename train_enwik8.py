from gpt_neox import (GPTNeoX, AutoregressiveWrapper, TextSamplerDataset,
                      cycle, prepare_optimizer_parameters, decode_tokens, prepare_enwik8_data)

import random
import torch
from torch.utils.data import DataLoader
import deepspeed

from tqdm.auto import trange
from gpt_neox.utils import GPUMonitor
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='GPTNeox Deepspeed Training Script')
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


# constants
NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 100
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 1024

# instantiate GPT-like decoder model
model = GPTNeoX(
    num_tokens=256,
    dim=512,
    seq_len=SEQ_LEN,
    depth=6,
    heads=8,
    dim_head=64
)

model = AutoregressiveWrapper(model)

# prepare enwik8 data
data_train, data_val = prepare_enwik8_data()
train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))
val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE))

# optimizer
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training
model_params = prepare_optimizer_parameters(model)
train_args = get_args()

# deepspeed loader
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
    pbar.set_description(f'Training Loss: {loss.item():.4f}')
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
        pbar.write(f"{prime} \n\n {'*' * 100}")
        sample = model.generate(inp, GENERATE_LENGTH)
        output_str = decode_tokens(sample)
        pbar.write(output_str)
