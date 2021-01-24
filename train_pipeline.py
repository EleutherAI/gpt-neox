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
from gpt_neox.datasets import GPT2Dataset
from gpt_neox.data_utils import get_tokenizer
from gpt_neox.utils import is_main, get_args, get_params
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

    # tokenizer
    tokenizer = get_tokenizer(tokenizer_type=params["tokenizer"].get("type", None),
                            from_pretrained=params["tokenizer"].get("from_pretrained", True),
                            add_padding_token=params["tokenizer"].get("add_padding_token", False))
    vocab_size = len(tokenizer) if params["vocab_size"] is None else params["vocab_size"]

    # model
    model = GPTNeoX_Pipe(
        num_tokens=vocab_size,
        dim=params["hidden_dim"],
        seq_len=params["seq_len"],
        depth=params["n_layers"],
        heads=params["n_heads"],
        dim_head=params["dim_head"],
        loss_fn = loss_function,
        num_stages = params.get("pipeline_num_stages", 2),
        activation_checkpoint_interval=params.get('activation_checkpoint_interval', 1)
    )

    # prepare data
    dset_params = params["dataset"]
    prepare_dataset(dset_params, train_args)

    train_dataset = GPT2Dataset(glob_pattern=dset_params["train_path"],
                                seq_len=params["seq_len"],
                                train=True,
                                mode='with_labels',
                                **dset_params)

    eval_dataset = GPT2Dataset(glob_pattern=dset_params["eval_path"],
                            seq_len=params["seq_len"],
                            train=False,
                            mode='with_labels',
                            **dset_params)

    val_loader = DataLoader(eval_dataset, batch_size=params["eval_batch_size"])
    val_loader = cycle(val_loader)

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
