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
from gpt_neox.utils import is_main, get_args, get_params
from gpt_neox.data_utils import read_enwik8_data
import gpt_neox

WORLD_SIZE = os.getenv('WORLD_SIZE')

def loss_function(x, y):
    losses = torch.nn.functional.cross_entropy(x, y, reduction='none')
    loss = losses.mean()
    return loss

def configure_checkpointing(model_engine):
    deepspeed.checkpointing.configure(model_engine.mpu, deepspeed_config=train_args.deepspeed_config)
    model_engine.mpu.checkpoint = deepspeed.checkpointing.checkpoint
    model_engine.mpu.get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
    model_engine.mpu.model_parallel_cuda_manual_seed = deepspeed.checkpointing.model_parallel_cuda_manual_seed
    assert deepspeed.checkpointing.is_configured()

def prepare_dataset(dset_params, train_args):
    torch.distributed.barrier()  # barrier will force processes to stop until *all* processes have reached the barrier
    if is_main(train_args):
        prepare_data(dset_params["name"])
        torch.distributed.barrier()  # barrier will force processes to stop until *all* processes have reached the barrier
    else:
        torch.distributed.barrier()


if __name__ == '__main__':
    # arguments
    train_args = get_args()
    params = get_params(train_args.model)
    deepspeed.init_distributed(dist_backend='nccl')
    model = gpt_neox.GPTNeoX_Pipe(
        num_tokens=params["vocab_size"],
        dim=params["hidden_dim"],
        seq_len=params["seq_len"],
        depth=params["n_layers"],
        heads=params["n_heads"],
        dim_head=params["dim_head"],
        loss_fn = loss_function,
        num_stages = params.get("pipeline_num_stages", 2),
        activation_checkpoint_interval=params.get('activation_checkpoint_interval', 1)
    )

    # prepare enwik8 data
    dset_params = params["dataset"]
    prepare_dataset(dset_params, train_args)
    data_train, data_val = read_enwik8_data(dset_params["path"])
    train_dataset = TextSamplerDataset(data_train, params["seq_len"], mode="with_labels")
    val_dataset = TextSamplerDataset(data_val, params["seq_len"], mode="with_labels")
    val_loader = cycle(DataLoader(val_dataset, batch_size=params["batch_size"]))

    # optimizer
    ds_model_params = prepare_optimizer_parameters(model)
    optim = torch.optim.Adam(ds_model_params, lr=params["learning_rate"])
    # deepspeed loader
    model_engine, optim, train_loader, _ = deepspeed.initialize(args=train_args,
                                                                model=model,
                                                                optimizer=optim,
                                                                model_parameters=ds_model_params,
                                                                training_data=train_dataset)
    configure_checkpointing(model_engine)

    batches_to_train = 10000

    pbar = trange(batches_to_train, mininterval=10., desc='Training Model', dynamic_ncols=True)
    for _ in pbar:
        for i in range(batches_to_train):

            loss = model_engine.train_batch()
            pbar.set_description(f'Training Loss: {loss.item():.4f}')
            pbar.update()
