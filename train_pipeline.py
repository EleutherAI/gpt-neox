import argparse
import json
import random
from collections import defaultdict
import os
import deepspeed
import torch
from torch.utils.data import DataLoader
from tqdm.auto import trange

from gpt_neox import (GPTNeoX, AutoregressiveWrapper, TextSamplerDataset,
                      cycle, prepare_optimizer_parameters, decode_tokens, prepare_data,
                      GPTNeoX_Pipe)
from gpt_neox.utils import is_main
import gpt_neox

WORLD_SIZE = os.getenv('WORLD_SIZE')

# arguments
train_args = get_args()
params = get_params(train_args.model)

# tokenizer
tokenizer = get_tokenizer(tokenizer_type=params["tokenizer"].get("type", None),
                          from_pretrained=params["tokenizer"].get("from_pretrained", True),
                          add_padding_token=params["tokenizer"].get("add_padding_token", False))
vocab_size = len(tokenizer) if params["vocab_size"] is None else params["vocab_size"]

# model
deepspeed.init_distributed(dist_backend='nccl')
torch.distributed.barrier()  # barrier will force processes to stop until *all* processes have reached the barrier

def loss_function(x, y):
    losses = torch.nn.functional.cross_entropy(x, y, reduction='none')
    loss = losses.mean()
    return loss
        
model = GPTNeoX_Pipe(
    num_tokens=params["vocab_size"],
    dim=params["hidden_dim"],
    seq_len=params["seq_len"],
    depth=params["n_layers"],
    heads=params["n_heads"],
    dim_head=params["dim_head"],
    loss_fn = loss_function,#torch.nn.CrossEntropyLoss(),
    num_stages = params.get("pipeline_num_stages", 2)
)
model = AutoregressiveWrapper(model)

# optimizer
ds_model_params = prepare_optimizer_parameters(model)
optim = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

# prepare data
dset_params = params["dataset"]
assert dset_params is not None

if is_main(train_args):
    prepare_data(dset_params["name"])
    torch.distributed.barrier()  # barrier will force processes to stop until *all* processes have reached the barrier
else:
    torch.distributed.barrier()
    
# data loading
train_dataset = GPT2Dataset(glob_pattern=dset_params["train_path"],
                            seq_len=params["seq_len"],
                            train=True,
                            **dset_params)
train_loader = model_engine.deepspeed_io(train_dataset, pin_memory=params.get("pin_memory", False))

eval_dataset = GPT2Dataset(glob_pattern=dset_params["eval_path"],
                           seq_len=params["seq_len"],
                           train=False,
                           **dset_params)

val_loader = DataLoader(eval_dataset, batch_size=params["eval_batch_size"])
val_loader = iter(val_loader)

# deepspeed loader
model_engine, optim, train_loader, _ = deepspeed.initialize(args=train_args,
                                                            model=model,
                                                            optimizer=optim,
                                                            model_parameters=ds_model_params,
                                                            training_data=train_dataset)


batches_to_train = 10000

pbar = trange(params["num_epochs"], mininterval=10., desc='Training Model', dynamic_ncols=True)
for _ in pbar:
    for i in range(batches_to_train):

        is_main = model_engine.local_rank == 0

        loss = model_engine.train_batch()

        pbar.set_description(f'Training Loss: {loss.item():.4f}')
        pbar.update()
