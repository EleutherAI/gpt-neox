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

def do_forward_pass(neox_args, model, inference=False):
    
    # set to eval mode
    model_was_in_train = model.training
    model.eval()
    
    # get context tokens
    # always forward full batch size
    context_tokens_tensor = torch.arange(2049).repeat((neox_args.train_micro_batch_size_per_gpu, 1)).cuda()

    # forward
    if inference:
        tokens, attention_mask, position_ids = get_batch(neox_args, context_tokens_tensor[:, :2048])
        model_inputs = (tokens,
                        position_ids,
                        attention_mask,
                        torch.Tensor(),
                        )
        logits, _ = forward_model(neox_args, model, model_inputs)
    elif neox_args.is_pipe_parallel:
        data_iterator = iter([{"text": context_tokens_tensor}])
        _, logits = model.eval_batch(data_iter=data_iterator, return_logits=True)
    else:
        tokens, attention_mask, position_ids = get_batch(neox_args, context_tokens_tensor[:, :2048])
        logits = model((tokens, position_ids, attention_mask))

    # reset to train mode, if model was in training before
    if model_was_in_train:
        model.train()

    if logits is not None:
        logits = logits.detach().cpu()[0] # just return first batch item (they are all equal)

    return logits

def check_forward_pass(neox_args, model, checkpoint_logits, inference):
    # do forward pass with loaded checkpoint
    logits = do_forward_pass(neox_args=neox_args, model=model, inference=inference)

    # check
    if logits is not None and checkpoint_logits is not None: # this could be the case for non-final pipeline stages
        if not (logits == checkpoint_logits).all().item():
            if mpu.get_data_parallel_rank() == 0:
                    print(" > WARNING: validate_checkpoint_forward() forward after load of checkpoint does not yield exactly same result")
            assert torch.isclose(logits, checkpoint_logits).all().item(), "validate_checkpoint_forward() forward after load of checkpoint does not yield a close result"

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
        logits = do_forward_pass(neox_args=neox_args, model=model)
        sd['checkpoint_validation_logits'] = logits
    
    model.save_checkpoint(neox_args.save, client_state=sd)

def save_checkpoint(neox_args, iteration, model, optimizer, lr_scheduler):
    """Save a model checkpoint."""

    if neox_args.do_distillation:
        _ = save_checkpoint_student(model, neox_args.load_teacher, neox_args.save, "distillGPTNeo")

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

def get_path(load):
    if "latest" in os.listdir(load):
        with open(os.path.join(load, "latest"), "r") as f:
            latest_folder = f.read()
        load = os.path.join(load, latest_folder)
    return load

def get_layer_number(filename):
    # layer_21-model_00-model_states.pt
    return int(filename.split("-")[0].split("_")[1])

def remove_mp_rank_ckpt(checkpoint_files):
    layer_ckpts = []; mp_rank_ckpt= None
    for ckpt_file in checkpoint_files:
        if "mp_rank" in ckpt_file:
            mp_rank_ckpt = ckpt_file
        else:
            layer_ckpts.append(ckpt_file)
    return sorted(layer_ckpts), mp_rank_ckpt 

def save_checkpoint_student(model, load_teacher, save_dir, tag):

    print_rank_0(f"Extracting and saving student model in {save_dir} as checkpoint from distil model")

    model.save_checkpoint(save_dir=save_dir, tag=tag)
    load_dir = os.path.join(save_dir, tag)

    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:

            pt_files, _ = remove_mp_rank_ckpt(os.listdir(get_path(load_dir)))
            teacher_pt_files, _ = remove_mp_rank_ckpt(os.listdir(get_path(load_teacher)))
            teacher_last_layer = get_layer_number(teacher_pt_files[-1]) + 1
            leading_zeros = len(str(teacher_last_layer))

            for pt_file in pt_files:
                layer_num = get_layer_number(pt_file)
                if layer_num < teacher_last_layer:
                    os.remove(os.path.join(load_dir, pt_file))
                else:
                    new_layer_num = layer_num - teacher_last_layer
                    new_pt_file = f"layer_{str(new_layer_num).zfill(leading_zeros)}-model_00-model_states.pt"
                    os.rename(os.path.join(load_dir, pt_file), os.path.join(load_dir, new_pt_file))
    
    torch.distributed.barrier()
    return load_dir

def combine_checkpoints(model, load_teacher, load_student, save_dir):

    teacher_checkpoint_files, _ = remove_mp_rank_ckpt(os.listdir(get_path(load_teacher)))
    teacher_last_layer = get_layer_number(teacher_checkpoint_files[-1]) + 1

    load_student = load_student if load_student is not None \
                else save_checkpoint_student(model, load_teacher, save_dir, tag="temp_stud_model")

    student_checkpoint_files, _ = remove_mp_rank_ckpt(os.listdir(get_path(load_student)))
    student_last_layer = get_layer_number(student_checkpoint_files[-1]) + 1

    load_dir = os.path.join(save_dir, "temp")

    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:

            if not os.path.exists(load_dir):
                os.makedirs(load_dir)

            print_rank_0(f"Creating temp checkpoint for distillation using {load_teacher}, {load_student} at {load_dir}")
            
            leading_zeros = len(str(teacher_last_layer+student_last_layer))
            
            for model_number in [0,1]:
                checkpoint_files = teacher_checkpoint_files if model_number==0 else student_checkpoint_files
                for ckpt_filename in checkpoint_files:
                    layer_num = get_layer_number(ckpt_filename)
                    model_dir = get_path(load_teacher) if model_number==0 else get_path(load_student)
                    new_layer_num = layer_num if model_number==0 else layer_num + teacher_last_layer
                    new_ckpt_filename = f"layer_{str(new_layer_num).zfill(leading_zeros)}-model_00-model_states.pt"
                    shutil.copy(os.path.join(model_dir, ckpt_filename), os.path.join(load_dir, new_ckpt_filename))

            dp_world_size = torch.distributed.get_world_size() if mpu is None else mpu.get_data_parallel_world_size()
            mp_world_size = 1 if mpu is None else mpu.get_model_parallel_world_size()

            client_sd = {'module': None, 
                        'optimizer': None, 
                        'lr_scheduler': None, 
                        'csr_tensor_module_names': set(), 
                        'skipped_steps': 0, 
                        'global_steps': 0, 
                        'global_samples': 0, 
                        'dp_world_size': dp_world_size, 
                        'mp_world_size': mp_world_size}

            torch.save(client_sd, os.path.join(load_dir, "mp_rank_00_model_states.pt"))

    torch.distributed.barrier()
    return save_dir, "temp"

def load_checkpoint(neox_args, model, optimizer, lr_scheduler, inference=False):
    """Load a model checkpoint and return the iteration."""

    if neox_args.deepspeed:
        load_optim_and_scheduler = not neox_args.no_load_optim  # TODO: These should be configured by separate args
        load_module_strict = True
        tag = None

        if neox_args.do_distillation is not None and neox_args.load is None:
            load_module_strict = False
            load_optim_and_scheduler = False
            if neox_args.load_teacher is not None:
                neox_args.load, tag = combine_checkpoints(model, neox_args.load_teacher, neox_args.load_student, neox_args.save)
            else:
                raise ValueError('Please provide path of the teacher model checkpoint')

        checkpoint_name, state_dict = model.load_checkpoint(neox_args.load,
                                                            load_optimizer_states=load_optim_and_scheduler,
                                                            load_lr_scheduler_states=load_optim_and_scheduler,
                                                            load_module_strict=load_module_strict,
                                                            tag=tag)

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

    # Check loaded checkpoint with forward pass
    if neox_args.checkpoint_validation_with_forward_pass:
        if "checkpoint_validation_logits" in state_dict:
            check_forward_pass(
                neox_args=neox_args, 
                model=model, 
                checkpoint_logits=state_dict["checkpoint_validation_logits"],
                inference=inference
                )
            print_rank_0(' > validated loaded checkpoint with forward pass ...')
        else:
            if mpu.get_data_parallel_rank() == 0:
                print(' > WARNING: checkpoint_validation_with_forward_pass is configured but no checkpoint validation data available in checkpoint {}'.format(checkpoint_name))

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
    if mpu.get_data_parallel_rank() == 0:
        print('  successfully loaded {}'.format(checkpoint_name))

    return iteration
