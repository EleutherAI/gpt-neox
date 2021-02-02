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

from gpt_neox.utils import is_main, get_args, save_ds_checkpoint, load_ds_checkpoint

import gpt_neox

WORLD_SIZE = os.getenv('WORLD_SIZE')

def loss_function(x, y):
    losses = torch.nn.functional.cross_entropy(x, y, reduction='none')
    loss = losses.mean()
    return loss

def configure_checkpointing(model_engine):
    deepspeed.checkpointing.configure(model_engine.mpu, deepspeed_config=args)
    model_engine.mpu.checkpoint = deepspeed.checkpointing.checkpoint
    model_engine.mpu.get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
    model_engine.mpu.model_parallel_cuda_manual_seed = deepspeed.checkpointing.model_parallel_cuda_manual_seed
    assert deepspeed.checkpointing.is_configured()

def prepare_dataset(dset_params, args):
    torch.distributed.barrier()  # barrier will force processes to stop until *all* processes have reached the barrier
    if is_main(args):
        prepare_data(dset_params["name"])
        torch.distributed.barrier()  # barrier will force processes to stop until *all* processes have reached the barrier
    else:
        torch.distributed.barrier()

if __name__ == '__main__':
    # arguments
    args = get_args()

    IS_MAIN = is_main(args)

    deepspeed.init_distributed(dist_backend='nccl')

    # tokenizer
    tokenizer = get_tokenizer(tokenizer_type=args["tokenizer"].get("type", None),
                            from_pretrained=args["tokenizer"].get("from_pretrained", True),
                            add_padding_token=args["tokenizer"].get("add_padding_token", False))
    vocab_size = len(tokenizer) if args["vocab_size"] is None else args["vocab_size"]

    # model
    model = GPTNeoX_Pipe(
        num_tokens=vocab_size,
        dim=args["hidden_dim"],
        seq_len=args["seq_len"],
        depth=args["n_layers"],
        heads=args["n_heads"],
        dim_head=args["dim_head"],
        loss_fn = loss_function,
        num_stages = args.get("pipeline_num_stages", 2),
        activation_checkpoint_interval=args.get('activation_checkpoint_interval', 1)
    )

    # prepare data
    dset_params = args["dataset"]
    prepare_dataset(dset_params, args)

    train_dataset = GPT2Dataset(glob_pattern=dset_params["train_path"],
                                seq_len=args["seq_len"],
                                train=True,
                                mode='with_labels',
                                **dset_params)

    eval_dataset = GPT2Dataset(glob_pattern=dset_params["eval_path"],
                            seq_len=args["seq_len"],
                            train=False,
                            mode='with_labels',
                            **dset_params)

    val_loader = DataLoader(eval_dataset, batch_size=args["eval_batch_size"])
    val_loader = cycle(val_loader)

    # optimizer
    ds_model_params = prepare_optimizer_parameters(model)
    optim = torch.optim.Adam(ds_model_params, lr=args["learning_rate"])
    # deepspeed loader

    model, optim, train_loader, lr_scheduler = deepspeed.initialize(args=args,
                                                                    model=model,
                                                                    optimizer=optim,
                                                                    model_parameters=ds_model_params,
                                                                    training_data=train_dataset)
    configure_checkpointing(model)
    current_iteration = load_ds_checkpoint(model, args, iteration=None)
    pbar = trange(current_iteration, args.get('train_steps', 100000), mininterval=10., desc='Training Model', dynamic_ncols=True)
    for i in pbar:
        loss = model.train_batch()
        pbar.set_description(f'Training Loss: {loss.item():.4f}')
        pbar.update()
        if not i % args.get('checkpoint_save_frequency', 1000) and i != 0:
            save_ds_checkpoint(i, model, args, args.get('keep_n_latest_checkpoints', 5), IS_MAIN)
