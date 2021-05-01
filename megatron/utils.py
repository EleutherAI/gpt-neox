# coding=utf-8

# Copyright (c) 2021 Josh Levy-Kramer <josh@levykramer.co.uk>.
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""General utilities."""
import os
import sys
import re
import time
import socket
from typing import Dict, List

import requests
import wandb
from wandb import UsageError

import torch

from deepspeed.launcher.runner import fetch_hostfile, parse_inclusion_exclusion

from megatron import print_rank_0
from megatron import mpu
from megatron.data.samplers import DistributedBatchSampler
from deepspeed import PipelineEngine, DeepSpeedEngine


def reduce_losses(losses):
    """Reduce a tensor of losses across all GPUs."""
    reduced_losses = torch.cat(
        [loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(reduced_losses)
    reduced_losses = reduced_losses / torch.distributed.get_world_size()
    return reduced_losses


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(torch.cuda.memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(
        torch.cuda.max_memory_reserved() / mega_bytes)
    print_rank_0(string)


def check_adlr_autoresume_termination(neox_args, iteration, model, optimizer, lr_scheduler):
    """Check for autoresume signal and exit if it is received."""
    # to prevent circular import
    from megatron.checkpointing import save_checkpoint

    # Add barrier to ensure consistnecy.
    torch.distributed.barrier()
    if neox_args.adlr_autoresume_object.termination_requested():
        if neox_args.save:
            save_checkpoint(neox_args=neox_args, iteration=iteration, model=model, optimizer=optimizer,
                            lr_scheduler=lr_scheduler)
        print_rank_0(">>> autoresume termination request found!")
        if torch.distributed.get_rank() == 0:
            neox_args.adlr_autoresume_object.request_resume()
        print_rank_0(">>> training terminated. Returning")
        sys.exit(0)


def make_data_loader(dataset, neox_args):
    """Buld dataloader given an input dataset."""
    if dataset is None:
        return None
    # Data parallel arguments.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    global_batch_size = neox_args.batch_size * world_size
    num_workers = neox_args.num_workers

    # Use a simple sampler with distributed batch sampler.
    sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = DistributedBatchSampler(sampler=sampler,
                                            batch_size=global_batch_size,
                                            drop_last=True,
                                            rank=rank,
                                            world_size=world_size)
    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=num_workers,
                                       pin_memory=True)


def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
        att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)

    return attention_mask, loss_mask, position_ids


def local_rank():
    """ Local rank of process """
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        print("utils.local_rank() environment variable LOCAL_RANK not set, defaulting to 0", flush=True)
        local_rank = 0
    return int(local_rank)


def is_local_main():
    """ True if is the local main process """
    return local_rank() == 0


def is_mp_rank_0():
    """True if mp rank == 0"""
    return mpu.get_model_parallel_rank() == 0


def get_wandb_api_key():
    """ Get Weights and Biases API key from ENV or .netrc file. Otherwise return None """
    if 'WANDB_API_KEY' in os.environ:
        return os.environ['WANDB_API_KEY']

    wandb_token = requests.utils.get_netrc_auth('https://api.wandb.ai')

    if wandb_token is not None:
        return wandb_token[1]


def init_wandb(neox_args):
    # Wandb. (one worker per machine)
    if neox_args.use_wandb == False:
        return

    use_wandb = is_local_main() and (get_wandb_api_key() is not None)
    neox_args.update_value("use_wandb", use_wandb)
    if neox_args.use_wandb:
        group_name = neox_args.wandb_group
        name = f'{socket.gethostname()}-{local_rank()}' if group_name else None
        try:
            wandb.init(project="neox", group=group_name, name=name, save_code=False,
                       force=False, entity=neox_args.wandb_team)
        except UsageError as e:
            neox_args.update_value("use_wandb", False)
            print(e)
            print('Skipping wandb. Execute `wandb login` on local or main node machine to enable.', flush=True)
        wandb.config.update(neox_args.all_config)


def obtain_resource_pool(hostfile_path, include_arg, exclude_arg) -> Dict[str, List[int]]:
    """
    Get dict of `resource_pool[hostname] = [list of GPU ranks]` using hostfile, include and exclude args.
    Modified from: `deepspeed.launcher.runner.main`
    """
    resource_pool = fetch_hostfile(hostfile_path)
    if not resource_pool:
        resource_pool = {}
        device_count = torch.cuda.device_count()
        if device_count == 0:
            raise RuntimeError("Unable to proceed, no GPU resources available")
        resource_pool['localhost'] = device_count

    active_resources = parse_inclusion_exclusion(resource_pool,
                                                 include_arg,
                                                 exclude_arg)
    return active_resources


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def tb_wandb_log(key, value, iteration_no, use_wandb, tensorboard_writer=None):
    # logs to both tb and wandb (if present) from the zeroth rank
    if torch.distributed.get_rank() == 0:
        if tensorboard_writer:
            tensorboard_writer.add_scalar(key, value, iteration_no)
        if use_wandb:
            wandb.log({key: value}, step=iteration_no)


def ddb(rank=0):
    """
    Distributed Debugger that will insert a py debugger on rank `rank` and
    pause all other distributed processes until debugging is complete.
    :param rank:
    """
    if torch.distributed.get_rank() == rank:
        from pdb import Pdb
        pdb = Pdb(skip=["torch.distributed.*"])
        pdb.set_trace(sys._getframe().f_back)
    torch.distributed.barrier()


class Timer:
    """Timer."""

    def __init__(self, name):
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()

    def start(self):
        """Start the timer."""
        assert not self.started_, 'timer has already been started'
        torch.cuda.synchronize()
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        """Stop the timer."""
        assert self.started_, 'timer is not started'
        torch.cuda.synchronize()
        self.elapsed_ += (time.time() - self.start_time)
        self.started_ = False

    def reset(self):
        """Reset timer."""
        self.elapsed_ = 0.0
        self.started_ = False

    def elapsed(self, reset=True):
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self.elapsed_
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if started_:
            self.start()
        return elapsed_


class Timers:
    """Group of timers."""

    def __init__(self, use_wandb, tensorboard_writer):
        self.timers = {}
        self.use_wandb = use_wandb
        self.tensorboard_writer = tensorboard_writer

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = Timer(name)
        return self.timers[name]

    def write(self, names, iteration, normalizer=1.0, reset=False):
        """Write timers to a tensorboard writer"""
        # currently when using add_scalars,
        # torch.utils.add_scalars makes each timer its own run, which
        # polutes the runs list, so we just add each as a scalar
        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed(reset=reset) / normalizer

            if self.tensorboard_writer:
                self.tensorboard_writer.add_scalar(f"timers/{name}", value, iteration)

            if self.use_wandb:
                wandb.log({f"timers/{name}": value}, step=iteration)

    def log(self, names, normalizer=1.0, reset=True):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = 'time (ms)'
        for name in names:
            elapsed_time = self.timers[name].elapsed(
                reset=reset) * 1000.0 / normalizer
            string += ' | {}: {:.2f}'.format(name, elapsed_time)
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                print(string, flush=True)
        else:
            print(string, flush=True)


def expand_attention_types(attention_config, num_layers):
    """
    Expands an `attention_config` list in the following format:

        [
        [['attention_type_1', ..., `attention_type_n`], 12]
        ]

    to a flattened list of length `num_layers`.

    :param params_list:
    :return:
    """
    # if only strings are found in the config, we assume it's already expanded
    if all([isinstance(i, str) for i in attention_config]):
        return attention_config
    newlist = []
    for item in attention_config:
        # instead of specifying a number - we can specify 'all' to extend this pattern across all layers
        if item[1] == "all":
            assert num_layers % len(item[0]) == 0, f"Number of layers ({num_layers}) is not divisible by the length " \
                                                   f"of pattern: {item[0]}"
            return item[0] * (num_layers // len(item[0]))
        for _ in range(item[1]):
            newlist.extend(item[0])
    return newlist
