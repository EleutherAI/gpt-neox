# Copyright (c) 2021, EleutherAI contributors
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

"""Pretrain GPT2"""
import socket

import torch
import wandb
from wandb import UsageError

from megatron.neox_arguments import NeoXArgs
from megatron import get_timers
from megatron import get_tokenizer
from megatron import mpu
from megatron import print_rank_0
from megatron.data.gpt2_dataset import build_train_valid_test_datasets
from megatron.fp16 import fp32_to_fp16
from megatron.global_vars import set_use_wandb
from megatron.model import GPT2ModelPipe
from megatron.model.gpt2_model import cross_entropy
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids, is_local_main, local_rank, get_wandb_api_key
from megatron.utils import reduce_losses


def init_wandb(use_wandb, args):
    # Wandb. (one worker per machine)
    use_wandb = is_local_main() and (get_wandb_api_key() is not None) and use_wandb
    set_use_wandb(use_wandb)
    args_dict = vars(args)
    if use_wandb:
        group_name = args_dict.get('wandb_group')
        name = f'{socket.gethostname()}-{local_rank()}' if group_name else None
        try:
            wandb.init(project="neox", group=group_name, name=name, save_code=False,
                       force=False, entity=args_dict.get('wandb_team'))
        except UsageError as e:
            set_use_wandb(False)
            print(e)
            print('Skipping wandb. Execute `wandb login` on local or main node machine to enable.')
        wandb.config.update(args_dict)


def model_provider(use_wandb=True, inference=False, get_key_value=True):
    """Build the model."""

    args = get_args() # TODO remove_global_vars
    print_rank_0('building GPT2 model ...')
    model = GPT2ModelPipe(num_tokentypes=0, parallel_output=True, topology=mpu.get_topology(), inference=inference,
                          get_key_value=get_key_value)
    if not args.is_pipe_parallel:
        # Export PipeParallel model to nn.Sequential model to avoid the overhead of deepspeed's pipe parallel training
        model = model.to_sequential()
    else:
        # This is a hack to give us a reference to get_batch_pipe from within training.py
        # We need to call model.set_batch_fn after deepspeed.initialize
        model._megatron_batch_fn = get_batch_pipe
    init_wandb(use_wandb, args)
    return model


def _get_batch(args, tokenizer, keys, data, datatype):
    """Support function for get_batch / get_batch pipe (to avoid code repetition)"""
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args() # TODO remove_global_vars
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    return _get_batch(args, tokenizer, keys, data, datatype)


def get_batch_pipe(data):
    """A modification of get_batch() to work with the latest batch instead of an iterator. """
    args = get_args() # TODO remove_global_vars
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    tokens, labels, loss_mask, attention_mask, position_ids = _get_batch(args, tokenizer, keys, data, datatype)
    # unpack data
    if args.precision == "fp16":
        # cast to fp16 because pipeline parallelism skips the FP16 wrapper.
        return fp32_to_fp16((tokens, position_ids, attention_mask)), fp32_to_fp16((labels, loss_mask))
    else:
        return (tokens, position_ids, attention_mask), (labels, loss_mask)


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args() # TODO remove_global_vars
    timers = get_timers()

    # Get the batch.
    timers('batch generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch generator').stop()

    outputs = model((tokens, position_ids, attention_mask))
    loss = cross_entropy(outputs, (labels, loss_mask), _fp16=args.fp16_lm_cross_entropy)

    # Reduce loss for logging.
    reduced_loss = reduce_losses([loss])

    return loss, {'lm loss': reduced_loss[0]}


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args() # TODO remove_global_vars

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT2 ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup))
    print_rank_0("> finished creating GPT2 datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    # TODO remove_global_vars get NeoXArgs from command line
    neox_args = NeoXArgs.from_ymls(["configs/small.yml", "configs/local_setup.yml"])
    pretrain(train_valid_test_dataset_provider=train_valid_test_datasets_provider, model_provider=model_provider, forward_step_func=forward_step, neox_args=neox_args)
