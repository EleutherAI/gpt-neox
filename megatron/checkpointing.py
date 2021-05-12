# coding=utf-8
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

"""Input/output checkpointing."""

import os
import re
import shutil
import random
import sys
import numpy as np

import torch
from glob import glob

from megatron import mpu
from megatron import print_rank_0
from megatron.utils import natural_sort

from megatron.text_generation_utils import get_batch, forward_model

def check_checkpoint_args(neox_args, checkpoint_args):
    """Ensure fixed arguments for a model are the same for the input
    arguments and the one retreived frm checkpoint."""

    assert isinstance(checkpoint_args, dict), "args stored in checkpoint is a dict"
    for checkpoint_arg_name, checkpoint_arg_value in checkpoint_args.items():
        args_value = getattr(neox_args, checkpoint_arg_name)
        error_message = '{} value from checkpoint ({}) is not equal to the currently set argument value ({}).'.format(checkpoint_arg_name, checkpoint_arg_value, args_value)
        assert checkpoint_arg_value == args_value, error_message


def ensure_directory_exists(filename):
    """Build filename's path if it does not already exists."""
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_checkpoint_name(checkpoints_path, iteration,
                        release=False, mp_rank=None):
    """A unified checkpoint name."""
    if release:
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(iteration)
    return os.path.join(checkpoints_path, directory,
                        'mp_rank_{:02d}'.format(
                            mpu.get_model_parallel_rank() if mp_rank is None
                            else mp_rank),
                        'model_optim_rng.pt')

def delete_old_checkpoints(save_dir, n_to_keep):
    if torch.distributed.get_rank() == 0:
        ckpt_dir_regex = r'global_step[\d]*'
        if save_dir.endswith('/'):
            save_dir = save_dir.strip('/')
        all_ckpts = natural_sort([i for i in glob(f'{save_dir}/*') if os.path.isdir(i)
                                  and re.search(ckpt_dir_regex, i)])
        n_to_delete = len(all_ckpts) - n_to_keep
        if n_to_delete > 0:
            to_delete = all_ckpts[:n_to_delete]
            print(f"WARNING: Deleting old checkpoints: \n\t{', '.join(to_delete)}")
            for ckpt in to_delete:
                try:
                    shutil.rmtree(ckpt)
                except FileNotFoundError:
                    pass


def save_ds_checkpoint(iteration, model, neox_args):
    """Save a model checkpoint."""
    sd = {
        'iteration': iteration,
        'args': {
            'num_layers': neox_args.num_layers,
            'hidden_size': neox_args.hidden_size,
            'num_attention_heads': neox_args.num_attention_heads,
            'max_position_embeddings': neox_args.max_position_embeddings,
            'make_vocab_size_divisible_by': neox_args.make_vocab_size_divisible_by,
            'padded_vocab_size': neox_args.padded_vocab_size,
            'tokenizer_type': neox_args.tokenizer_type,
            'model_parallel_size': neox_args.model_parallel_size
            }
        }
    # rng states.
    if not neox_args.no_save_rng:
        sd['random_rng_state'] = random.getstate()
        sd['np_rng_state'] = np.random.get_state()
        sd['torch_rng_state'] = torch.get_rng_state()
        sd['cuda_rng_state'] = torch.cuda.get_rng_state()
        sd['rng_tracker_states'] = mpu.get_cuda_rng_tracker().get_states()
    
    if neox_args.checkpoint_validation_with_forward_pass:
        torch.distributed.barrier()

        # set to eval mode
        model_was_in_train = model.training
        model.eval()

        # forward
        context_tokens_tensor = torch.randint(0, neox_args.padded_vocab_size, (4, neox_args.seq_length + 1 )).to(torch.int64) 
        tokens, attention_mask, position_ids = get_batch(neox_args, context_tokens_tensor)
        if neox_args.setup_model_for_inference:
            model_inputs = (tokens, position_ids, attention_mask, torch.Tensor())
            logits, _ = forward_model(neox_args, model, model_inputs)
        else:
            #TODO add capabilities for pipeline parallel
            #if neox_args.pipe_parallel_size <= 1:
            model_inputs = (tokens, position_ids, attention_mask)
            #else:
            #    model_inputs = (tokens, position_ids, attention_mask, (torch.Tensor(1), torch.Tensor(1)))
            logits = forward_model(neox_args, model, model_inputs)
        torch.distributed.barrier()

        # add info to state
        sd["checkpoint_validation"] = {
            "context_token_tensor": context_tokens_tensor,
            "logits": logits.detach().cpu(),
        }
        
        # reset to train mode, if model was in training before
        if model_was_in_train:
            model.train()

        torch.distributed.barrier()

    
    model.save_checkpoint(neox_args.save, client_state=sd)

def validate_checkpoint_forward(neox_args, model, checkpoint_context_token_tensor, checkpoint_logits_tensor):
    torch.distributed.barrier()

    assert torch.is_tensor(checkpoint_context_token_tensor), "validate_checkpoint_forward() input checkpoint_context_token_tensor is tensor"
    assert torch.is_tensor(checkpoint_logits_tensor), "validate_checkpoint_forward() input checkpoint_logits_tensor is tensor"

    # set to eval mode
    model_was_in_train = model.training
    model.eval()

    # forward
    tokens, attention_mask, position_ids = get_batch(neox_args, checkpoint_context_token_tensor)
    if neox_args.setup_model_for_inference:
        model_inputs = (tokens, position_ids, attention_mask, torch.Tensor())
        logits, _ = forward_model(neox_args, model, model_inputs)
    else:
        #TODO add capabilities for pipeline parallel
        model_inputs = (tokens, position_ids, attention_mask)
        logits = forward_model(neox_args, model, model_inputs)
    torch.distributed.barrier()

    # check
    logits = logits.detach().cpu()
    is_exactly_equal = (logits == checkpoint_logits_tensor).all().item()
    is_close = torch.isclose(logits, checkpoint_logits_tensor).all().item()
    if not is_exactly_equal:
        if mpu.get_data_parallel_rank() == 0:
                print(" > WARNING: validate_checkpoint_forward() forward after load of checkpoint does not yield exactly same result")
        assert is_close, "validate_checkpoint_forward() forward after load of checkpoint does not yield a close result"

    # reset to train mode, if model was in training before
    if model_was_in_train:
        model.train()

    torch.distributed.barrier()

def save_checkpoint(neox_args, iteration, model, optimizer, lr_scheduler):
    """Save a model checkpoint."""

    if neox_args.deepspeed:
        save_ds_checkpoint(iteration, model, neox_args)
    else:
        raise ValueError('Must be using deepspeed to use neox')

    # Wait so everyone is done (necessary)
    torch.distributed.barrier()
    if neox_args.keep_last_n_checkpoints is not None:
        delete_old_checkpoints(neox_args.save, neox_args.keep_last_n_checkpoints)

    # Wait so everyone is done (not necessary)
    torch.distributed.barrier()


def load_checkpoint(neox_args, model, optimizer, lr_scheduler):
    """Load a model checkpoint and return the iteration."""

    if neox_args.deepspeed:
        load_optim_and_scheduler = not neox_args.no_load_optim  # TODO: These should be configured by separate args
        checkpoint_name, state_dict = model.load_checkpoint(neox_args.load,
                                                            load_optimizer_states=load_optim_and_scheduler,
                                                            load_lr_scheduler_states=load_optim_and_scheduler)

        if checkpoint_name is None:
            if mpu.get_data_parallel_rank() == 0:
                print("Unable to load checkpoint.")
            return 0 # iteration 0, if not checkpoint loaded
    else:
        raise ValueError('Must be using deepspeed to use neox')

    # Set iteration.
    if neox_args.finetune:
        iteration = 0
    else:
        iteration = state_dict.get('iteration') or state_dict.get("total_iters") # total_iters backward compatible with older checkpoints
        if iteration is None:
            raise ValueError('Unable to load iteration from checkpoint {}, exiting'.format(checkpoint_name))

    # Check arguments.
    if 'args' in state_dict:
        checkpoint_args = state_dict['args']
        check_checkpoint_args(neox_args=neox_args, checkpoint_args=checkpoint_args)
        print_rank_0(' > validated currently set args with arguments in the checkpoint ...')
    else:
        print_rank_0(' > could not find arguments in the checkpoint for validation...')

    # rng states.
    if not neox_args.finetune and not neox_args.no_load_rng:
        try:
            random.setstate(state_dict['random_rng_state'])
            np.random.set_state(state_dict['np_rng_state'])
            torch.set_rng_state(state_dict['torch_rng_state'])
            torch.cuda.set_rng_state(state_dict['cuda_rng_state'])
            mpu.get_cuda_rng_tracker().set_states(
                state_dict['rng_tracker_states'])
        except KeyError:
            print_rank_0('Unable to load optimizer from checkpoint {}. '
                         'Specify --no-load-rng or --finetune to prevent '
                         'attempting to load the optimizer state, '
                         'exiting ...'.format(checkpoint_name))
            sys.exit()
    
    
    torch.distributed.barrier()
    if neox_args.checkpoint_validation_with_forward_pass:
        if "checkpoint_validation" in state_dict:
            validate_checkpoint_forward(
                neox_args=neox_args, 
                model=model, 
                checkpoint_context_token_tensor=state_dict["checkpoint_validation"]["context_token_tensor"], 
                checkpoint_logits_tensor=state_dict["checkpoint_validation"]["logits"], 
                )
            print_rank_0(' > validated loaded checkpoint with forward pass ...')
        else:
            if mpu.get_data_parallel_rank() == 0:
                print(' > WARNING: checkpoint_validation_with_forward_pass is configured but no checkpoint validation data available in checkpoint {}'.format(checkpoint_name))



    torch.distributed.barrier()
    if mpu.get_data_parallel_rank() == 0:
        print('  successfully loaded {}'.format(checkpoint_name))

    return iteration
