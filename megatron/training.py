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
#
# This file has been modified from its original version
#

"""Pretrain utilities."""
from datetime import datetime
from functools import partial

import math
import sys

import torch

from megatron.utils import Timers, init_wandb
from megatron import print_rank_0

from megatron import mpu

from megatron.model import GPT2ModelPipe
from megatron.checkpointing import load_checkpoint, save_checkpoint
from megatron.data.gpt2_dataset import build_train_valid_test_datasets

from megatron.initialize import initialize_megatron
from megatron.learning_rates import AnnealingLR
from megatron.model import get_params_for_weight_decay_optimization
from megatron.utils import check_adlr_autoresume_termination
from megatron.utils import make_data_loader
from megatron.utils import report_memory
from megatron.utils import tb_wandb_log
from megatron.gradient_noise_scale import GradientNoiseScale

from megatron.fp16 import fp32_to_fp16
from megatron.model.gpt2_model import cross_entropy
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import reduce_losses

import deepspeed


def pretrain(neox_args):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the model.

    Arguments:
        neox_args: an instance of NeoXArgs containing the configuration for pretrain

    """
    # setup logging and timers
    init_wandb(neox_args=neox_args)
    timers = Timers(use_wandb=neox_args.use_wandb, tensorboard_writer=neox_args.tensorboard_writer)

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(neox_args=neox_args)

    # Model, optimizer, and learning rate.
    timers('model and optimizer').start()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(neox_args=neox_args, inference=False, get_key_value=True)
    timers('model and optimizer').stop()

    # Data stuff.
    timers('train/valid/test data iterators').start()
    train_data_iterator, valid_data_iterator, test_data_iterator = build_train_valid_test_data_iterators(neox_args=neox_args)
    timers('train/valid/test data iterators').stop()

    # Print setup timing.
    print_rank_0('done with setups ...')
    timers.log(['model and optimizer', 'train/valid/test data iterators'])
    print_rank_0('training ...')

    iteration = 0
    if neox_args.do_train and neox_args.train_iters > 0:
        iteration = train(
            neox_args=neox_args, 
            timers=timers,
            model=model, 
            optimizer=optimizer, 
            lr_scheduler=lr_scheduler,
            train_data_iterator=train_data_iterator, 
            valid_data_iterator=valid_data_iterator
            )

    if neox_args.do_valid:
        prefix = 'the end of training for val data'
        evaluate_and_print_results(
            neox_args=neox_args, 
            prefix=prefix, 
            forward_step_func=forward_step,
            data_iterator=valid_data_iterator, 
            model=model, 
            iteration=iteration, 
            verbose=False
            )

    if neox_args.save and iteration != 0:
        save_checkpoint(neox_args=neox_args, iteration=iteration, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)

    if neox_args.do_test:
        # Run on test data.
        prefix = 'the end of training for test data'
        evaluate_and_print_results(
            neox_args=neox_args, 
            prefix=prefix, 
            forward_step_func=forward_step,
            data_iterator=test_data_iterator, 
            model=model, 
            iteration=0, # iteration 0 in order to always use full test data
            verbose=True
            )

def _get_batch(neox_args, tokenizer, keys, data, datatype):
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
        neox_args.reset_position_ids,
        neox_args.reset_attention_mask,
        neox_args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids


def get_batch(neox_args, data_iterator):
    """Generate a batch"""

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    return _get_batch(neox_args=neox_args, tokenizer=neox_args.tokenizer, keys=keys, data=data, datatype=datatype)

def get_batch_pipe(data, neox_args):
    """A modification of get_batch() to work with the latest batch instead of an iterator. """
    
    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    tokens, labels, loss_mask, attention_mask, position_ids = _get_batch(neox_args, neox_args.tokenizer, keys, data, datatype)
    # unpack data
    if neox_args.precision == "fp16":
        # cast to fp16 because pipeline parallelism skips the FP16 wrapper.
        return fp32_to_fp16((tokens, position_ids, attention_mask)), fp32_to_fp16((labels, loss_mask))
    else:
        return (tokens, position_ids, attention_mask), (labels, loss_mask)


def forward_step(neox_args, timers, data_iterator, model):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(neox_args=neox_args, data_iterator=data_iterator)
    timers('batch generator').stop()

    outputs = model((tokens, position_ids, attention_mask))
    loss = cross_entropy(outputs, (labels, loss_mask), _fp16=neox_args.fp16_lm_cross_entropy)

    # Reduce loss for logging.
    reduced_loss = reduce_losses([loss])

    return loss, {'lm loss': reduced_loss[0]}

def get_model(neox_args, inference=False, get_key_value=True):
    """Build the model."""

    print_rank_0('building GPT2 model ...')

    # Build model on cpu.
    model = GPT2ModelPipe(neox_args=neox_args, num_tokentypes=0, parallel_output=True, topology=mpu.get_topology(), inference=inference, get_key_value=get_key_value)
    if not neox_args.is_pipe_parallel:
        # Export PipeParallel model to nn.Sequential model to avoid the overhead of deepspeed's pipe parallel training
        model = model.to_sequential()
    else:
        # This is a hack to give us a reference to get_batch_pipe from within training.py
        # We need to call model.set_batch_fn after deepspeed.initialize
        model._megatron_batch_fn = partial(get_batch_pipe, neox_args=neox_args) 

    if neox_args.deepspeed:
        # DeepSpeed handles CUDA, FP16, and DDP components.
        return model
    else:
        raise ValueError("Must be using deepspeed to run neox")


def get_optimizer(model, neox_args):
    """Set up the optimizer."""
    if neox_args.no_load_optim:
        return None, None
    # Build parameter groups (weight decay and non-decay).
    param_groups = get_params_for_weight_decay_optimization(model, neox_args)
    print_rank_0(f'Configuring Optimizer type: {neox_args.optimizer_type} with params: {neox_args.optimizer["params"]}')
    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        for param in param_group['params']:
            if not hasattr(param, 'model_parallel'):
                param.model_parallel = False

    if neox_args.optimizer_type.lower() in ["cpu_adam", "cpu_torch_adam"]:
        if neox_args.optimizer == "cpu_torch_adam":
            cpu_adam_optimizer = torch.optim.Adam
        else:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            cpu_adam_optimizer = DeepSpeedCPUAdam
        optimizer = cpu_adam_optimizer(param_groups,
                                       weight_decay=neox_args.weight_decay,
                                       **neox_args.optimizer["params"])
    elif neox_args.optimizer_type.lower() == "onebitadam":
        assert neox_args.deepspeed
        optimizer = None
        # onebitadam needs to be instantiated within the deepspeed engine to work :|
    elif neox_args.optimizer_type.lower() == "sm3":
        from .optimizers import SM3
        optimizer = SM3(
            param_groups,
            **neox_args.optimizer["params"])
    elif neox_args.optimizer_type.lower() == "madgrad_wd":
        from .optimizers import madgrad_wd
        optimizer = madgrad_wd(
            param_groups,
            weight_decay=neox_args.weight_decay,
            **neox_args.optimizer["params"])
    elif neox_args.optimizer_type.lower() == "adam":
        # Use Adam
        try:
            # default to apex as it's slightly faster
            from apex.optimizers import FusedAdam as Adam
        except ImportError:
            # if apex isn't installed, use deepspeed's FusedAdam
            print("WARNING: APEX not installed - defaulting to deepspeed's fused adam")
            from deepspeed.ops.adam import FusedAdam as Adam
        optimizer = Adam(param_groups,
                         weight_decay=neox_args.weight_decay,
                         **neox_args.optimizer["params"])
    else:
        raise ValueError(f"Optimizer type {neox_args.optimizer_type} not recognized")

    if neox_args.deepspeed:
        # fp16 wrapper is not required for DeepSpeed.
        return optimizer, param_groups
    else:
        raise ValueError("Must be using deepspeed to run neox")


def get_learning_rate_scheduler(optimizer, neox_args):
    """Build the learning rate scheduler."""
    if neox_args.no_load_optim:
        # TODO: this should be configured as a separate arg
        return None
    if neox_args.deepspeed and neox_args.optimizer_type.lower() == "onebitadam":
        print_rank_0("WARNING: onebitadam requires the lr scheduler be built by deepspeed - "
                     "Make sure one is added to your deepspeed config")
        return None

    # Add linear learning rate scheduler.
    if neox_args.lr_decay_iters is not None:
        num_iters = neox_args.lr_decay_iters
    else:
        num_iters = neox_args.train_iters
    num_iters = max(1, num_iters)
    init_step = 0
    warmup_iter = neox_args.warmup * num_iters
    lr_scheduler = AnnealingLR(
        optimizer,
        start_lr=neox_args.lr,
        warmup_iter=warmup_iter,
        total_iters=num_iters,
        decay_style=neox_args.lr_decay_style,
        last_iter=init_step,
        min_lr=neox_args.min_lr,
        use_checkpoint_lr_scheduler=neox_args.use_checkpoint_lr_scheduler,
        override_lr_scheduler=neox_args.override_lr_scheduler)

    return lr_scheduler


def setup_model_and_optimizer(neox_args, inference=False, get_key_value=True):
    """Setup model and optimizer."""
    model = get_model(neox_args=neox_args, inference=inference, get_key_value=get_key_value)
    optimizer, param_groups = get_optimizer(model=model, neox_args=neox_args)
    lr_scheduler = get_learning_rate_scheduler(optimizer=optimizer, neox_args=neox_args)

    if neox_args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")
        if neox_args.no_load_optim:
            assert optimizer is None
            _model_params = None
            _lr_scheduler = None
        else:
            _model_params = param_groups if optimizer is None else None
            _lr_scheduler = lr_scheduler

        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=neox_args,
            lr_scheduler=_lr_scheduler,
            dist_init_required=False,
            model_parameters=_model_params,
            config_params=neox_args.deepspeed_config,
            mpu=mpu if not neox_args.is_pipe_parallel else None
        )
        model.total_params = get_total_params(model.module)
        print_rank_0(f' > total params: {"{:,}".format(model.total_params)}')

        if neox_args.is_pipe_parallel:
            model.set_batch_fn(model.module._megatron_batch_fn)
    else:
        raise ValueError("Must be using deepspeed to run neox")

    if neox_args.load is not None:
        neox_args.iteration = load_checkpoint(neox_args=neox_args, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
        print_rank_0(f'Loading checkpoint and starting from iteration {neox_args.iteration}')
    else:
        neox_args.iteration = 0

    return model, optimizer, lr_scheduler


def backward_step(neox_args, timers, optimizer, model, loss):
    """Backward step."""

    # Backward pass.
    timers('backward-backward').start()
    if neox_args.deepspeed:
        model.backward(loss)
    else:
        raise ValueError("Must be using deepspeed to run neox")
    timers('backward-backward').stop()

    if neox_args.deepspeed:
        # DeepSpeed backward propagation already addressed all reduce communication.
        # Reset the timer to avoid breaking timer logs below.
        timers('backward-allreduce').reset()
    else:
        raise ValueError("Must be using deepspeed to run neox")


def train_step(neox_args, timers, data_iterator, model, optimizer, lr_scheduler):
    """Single training step."""

    # Pipeline parallelism schedules forward/backward/step
    if neox_args.is_pipe_parallel:
        return train_step_pipe(neox_args=neox_args, timers=timers, model=model, data_iterator=data_iterator)

    # Forward model for one step.
    timers('forward').start()
    loss, loss_reduced = forward_step(neox_args=neox_args, timers=timers, data_iterator=data_iterator, model=model)
    timers('forward').stop()

    # Calculate gradients, reduce across processes, and clip.
    timers('backward').start()
    backward_step(neox_args=neox_args, timers=timers, optimizer=optimizer, model=model, loss=loss)
    timers('backward').stop()

    # Update parameters.
    skipped_iter = 0
    timers('optimizer').start()
    if neox_args.deepspeed:
        model.step()
    else:
        raise ValueError("Must be using deepspeed to run neox")
    timers('optimizer').stop()

    return loss_reduced, skipped_iter


def train_step_pipe(neox_args, timers, model, data_iterator):
    """Single training step with DeepSpeed's pipeline parallel engine. """

    assert neox_args.deepspeed
    loss = model.train_batch(data_iter=data_iterator)
    loss_dict = {'lm loss': loss}
    if neox_args.precision == "fp16" and model.optimizer.overflow:
        skipped_iter = 1
    else:
        skipped_iter = 0
    # Don't break Megatron's timers because we changed code paths.
    for t in ['forward', 'backward', 'allreduce', 'optimizer', 'batch generator', 'data loader']:
        timers(t).reset()
    return loss_dict, skipped_iter


def training_log(neox_args, timers, loss_dict, total_loss_dict, learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter, model, optimizer, noise_scale_logger):
    """Log training information such as losses, timing, etc."""

    # Update losses.
    skipped_iters_key = 'skipped iterations'
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(
        skipped_iters_key, 0) + skipped_iter
    got_nan_key = 'got nan'

    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = total_loss_dict.get(key, 0.) + loss_dict[key]
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float('inf') or \
                     value == -float('inf') or \
                     value != value
            got_nan = got_nan or is_nan

    total_loss_dict[got_nan_key] = total_loss_dict.get(
        got_nan_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = []

    def add_to_logging(name):
        if name in timers.timers:
            timers_to_log.append(name)

    if not neox_args.is_pipe_parallel:
        add_to_logging('forward')
        add_to_logging('backward')
        add_to_logging('backward-backward')
        add_to_logging('backward-allreduce')
        add_to_logging('backward-master-grad')
        add_to_logging('backward-clip-grad')
        add_to_logging('optimizer')
        add_to_logging('batch generator')

        # Log timer info to tensorboard and wandb
        normalizer = iteration % neox_args.log_interval
        if normalizer == 0:
            normalizer = neox_args.log_interval
        if torch.distributed.get_rank() == 0:
            timers.write(names=timers_to_log, iteration=iteration, normalizer=normalizer)
    else:
        # with pipeline parallel, the megatron timers are overridden by the deepspeed ones.
        # Try to grab timer values from model engine. Only recently added to deeperspeed, so check that the engine
        # has that attribute first
        if hasattr(model, 'timer_values') and model.timer_values is not None:
            if model.wall_clock_breakdown() and model.global_steps % model.steps_per_print() == 0:
                timer_values = model.timer_values
                # deepspeed already logs to tensorboard / prints values, so just log to wandb
                if neox_args.use_wandb and torch.distributed.get_rank() == 0:
                    for key in timer_values:
                        tb_wandb_log(f"timers/{key}", timer_values[key], iteration, use_wandb=neox_args.use_wandb, tensorboard_writer=neox_args.tensorboard_writer)

    # write losses, lr, etc. every step
    tb_wandb_log('train/learning_rate', learning_rate, iteration, use_wandb=neox_args.use_wandb, tensorboard_writer=neox_args.tensorboard_writer)
    for key in loss_dict:
        tb_wandb_log(f'train/{key.replace(" ", "_")}', loss_dict[key], iteration, use_wandb=neox_args.use_wandb, tensorboard_writer=neox_args.tensorboard_writer)
    if neox_args.fp16:
        tb_wandb_log(f'train/loss_scale', loss_scale, iteration, use_wandb=neox_args.use_wandb, tensorboard_writer=neox_args.tensorboard_writer)

    # log gradient noise scale
    if neox_args.log_gradient_noise_scale:
        if noise_scale_logger.noise_scale is not None:
            tb_wandb_log(f'train/noise_scale', noise_scale_logger.noise_scale, iteration, use_wandb=neox_args.use_wandb, tensorboard_writer=neox_args.tensorboard_writer)

    # (optional) Log optimizer states to wandb / tb every step
    if neox_args.log_optimizer_states:
        for k, v in optimizer.state_dict()['optimizer_state_dict']['state'].items():
            for ki, vi in v.items():  # step, module
                if ki != 'step':
                    opt_state_norm = torch.norm(vi) if hasattr(vi, 'dim') else vi
                    tb_wandb_log(f'optimizer_state_norms/{k}_{ki}', opt_state_norm, iteration, use_wandb=neox_args.use_wandb, tensorboard_writer=neox_args.tensorboard_writer)

    # (optional) Log grad/param norms to wandb / tb every step
    if neox_args.log_grad_norm or neox_args.log_param_norm:
        for name, param in model.module.named_parameters():
            if neox_args.log_grad_norm:
                if param.grad is not None:
                    tb_wandb_log(f'gradient_norms/{name}', torch.norm(param.grad), iteration, use_wandb=neox_args.use_wandb, tensorboard_writer=neox_args.tensorboard_writer)
            if neox_args.log_param_norm:
                tb_wandb_log(f'parameter_norms/{name}', torch.norm(param), iteration, use_wandb=neox_args.use_wandb, tensorboard_writer=neox_args.tensorboard_writer)

    if iteration % neox_args.log_interval == 0:
        # log other stuff every neox_args.log_interval iters
        elapsed_time = timers('interval time').elapsed()
        iteration_time = elapsed_time / neox_args.log_interval
        samples_per_sec = get_global_batch_size(neox_args) / iteration_time
        log_string = ' samples/sec: {:.3f} |'.format(samples_per_sec)
        tb_wandb_log('runtime/samples_per_sec', samples_per_sec, iteration, use_wandb=neox_args.use_wandb, tensorboard_writer=neox_args.tensorboard_writer)
        tb_wandb_log('runtime/iteration_time', iteration_time, iteration, use_wandb=neox_args.use_wandb, tensorboard_writer=neox_args.tensorboard_writer)
        log_string += ' iteration {:8d}/{:8d} |'.format(iteration, neox_args.train_iters)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time * 1000.0 / neox_args.log_interval)
        log_string += ' learning rate: {:.3E} |'.format(learning_rate)
        num_iterations = max(
            1, neox_args.log_interval - total_loss_dict[skipped_iters_key])

        # log tflop / gpu
        flops_per_s_per_gpu = get_flops(neox_args=neox_args, model=model, iter_time_s=iteration_time)
        log_string += f' approx flops per GPU: {human_readable_flops(flops_per_s_per_gpu)} |'
        tb_wandb_log('runtime/flops_per_sec_per_gpu', flops_per_s_per_gpu, iteration, use_wandb=neox_args.use_wandb, tensorboard_writer=neox_args.tensorboard_writer)

        for key in total_loss_dict:
            if key not in [skipped_iters_key, got_nan_key]:
                v = total_loss_dict[key].item() if hasattr(total_loss_dict[key], 'item') else total_loss_dict[key]
                avg = v / float(num_iterations)
                log_string += ' {}: {:.6E} |'.format(key, avg)
                total_loss_dict[key] = 0.0
        if neox_args.precision == "fp16":
            log_string += ' loss scale: {:.1f} |'.format(loss_scale)
        log_string += ' number of skipped iterations: {:3d} |'.format(
            total_loss_dict[skipped_iters_key])
        log_string += ' number of nan iterations: {:3d} |'.format(
            total_loss_dict[got_nan_key])
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[got_nan_key] = 0
        print_rank_0(log_string)
        if report_memory_flag:
            report_memory('after {} iterations'.format(iteration))
            report_memory_flag = False

        timers.log(timers_to_log, normalizer=neox_args.log_interval)

    return report_memory_flag


def train(neox_args, timers, model, optimizer, lr_scheduler,
          train_data_iterator, valid_data_iterator):
    """Train the model function."""

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = neox_args.iteration

    timers('interval time').start()
    report_memory_flag = True

    if neox_args.log_gradient_noise_scale:
        if neox_args.zero_stage >= 1:
            raise NotImplementedError('Gradient Noise Scale logging does not work with zero stage 2+, as the '
                                      'gradients are distributed across ranks.')
        noise_scale_logger = GradientNoiseScale(
            model=model,
            batch_size_small=neox_args.train_batch_size,
            n_batches=neox_args.gradient_noise_scale_n_batches,
            cpu_offload=neox_args.gradient_noise_scale_cpu_offload,
            neox_args=neox_args,
            mpu=mpu)
    else:
        noise_scale_logger = None

    while iteration < neox_args.train_iters:
        loss_dict, skipped_iter = train_step(
            neox_args=neox_args, 
            timers=timers,
            data_iterator=train_data_iterator,
            model=model, 
            optimizer=optimizer, 
            lr_scheduler=lr_scheduler
            )
        iteration += 1
        if neox_args.log_gradient_noise_scale:
            noise_scale_logger.update()
        # Logging.
        loss_scale = None
        if neox_args.precision == "fp16":
            loss_scale = optimizer.cur_scale
        report_memory_flag = training_log(
            neox_args=neox_args, 
            timers=timers, 
            loss_dict=loss_dict, 
            total_loss_dict=total_loss_dict, 
            learning_rate=optimizer.param_groups[0]['lr'], 
            iteration=iteration,
            loss_scale=loss_scale, 
            report_memory_flag=report_memory_flag, 
            skipped_iter=skipped_iter, 
            model=model, 
            optimizer=optimizer, 
            noise_scale_logger=noise_scale_logger
            )

        # Autoresume
        if neox_args.adlr_autoresume and \
                (iteration % neox_args.adlr_autoresume_interval == 0):
            check_adlr_autoresume_termination(neox_args=neox_args, iteration=iteration, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)

        # Checkpointing
        if neox_args.save and neox_args.save_interval and iteration % neox_args.save_interval == 0:
            save_checkpoint(neox_args=neox_args, iteration=iteration, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)

        # Evaluation
        if neox_args.eval_interval and iteration % neox_args.eval_interval == 0 and neox_args.do_valid:
            prefix = 'iteration {}'.format(iteration)
            evaluate_and_print_results(
                neox_args=neox_args, 
                prefix=prefix, 
                forward_step_func=forward_step,
                data_iterator=valid_data_iterator, 
                model=model, 
                iteration=iteration, 
                verbose=False
                )

        if neox_args.exit_interval and iteration % neox_args.exit_interval == 0:
            torch.distributed.barrier()
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            rank = torch.distributed.get_rank()
            print_rank_0('rank: {} | time: {} | exiting the program at iteration {}'.format(rank, time_str, iteration))
            sys.exit()

    return iteration


def evaluate(neox_args, forward_step_fn, data_iterator, model, verbose=False):
    """Evaluation."""
    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss_dict = {}

    with torch.no_grad():
        iteration = 0
        while iteration < neox_args.eval_iters:
            iteration += 1
            if verbose and iteration % neox_args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration, neox_args.eval_iters))
            # Forward evaluation.
            _, loss_dict = forward_step_fn(data_iterator, model)

            # When contiguous memory optimizations are enabled, the buffers
            # allocated by the optimizations are deallocated during backward pass
            # in the absence of backward pass the buffers should be reset after each
            # forward pass
            if neox_args.deepspeed and neox_args.deepspeed_activation_checkpointing:
                deepspeed.checkpointing.reset()

            # Reduce across processes.
            for key in loss_dict:
                total_loss_dict[key] = total_loss_dict.get(key, 0.) + loss_dict[key]
    
    # Move model back to the train mode.
    model.train()

    for key in total_loss_dict:
        total_loss_dict[key] /= neox_args.eval_iters

    return total_loss_dict


def evaluate_and_print_results(neox_args, prefix, forward_step_func, data_iterator, model, iteration, verbose=False):
    """Helper function to evaluate and dump results on screen."""

    # Pipeline parallelism needs eval_batch() instead of a simple forward().
    if neox_args.is_pipe_parallel:
        def _eval_helper(data_iter, _):
            loss = model.eval_batch(data_iter)
            return None, {'lm loss': loss}
        forward_step_func = _eval_helper

    total_loss_dict = evaluate(neox_args=neox_args, forward_step_fn=forward_step_func, data_iterator=data_iterator, model=model, verbose=verbose)
    string = ' validation loss at {} | '.format(prefix)
    for key in total_loss_dict:
        string += '{} value: {:.6E} | '.format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += '{} PPL: {:.6E} | '.format(key, ppl)
        tb_wandb_log(f"validation/{key.replace(' ', '_')}", total_loss_dict[key].item(), iteration, use_wandb=neox_args.use_wandb, tensorboard_writer=neox_args.tensorboard_writer)
        tb_wandb_log(f"validation/{key.replace(' ', '_')}_ppl", ppl, iteration, use_wandb=neox_args.use_wandb, tensorboard_writer=neox_args.tensorboard_writer)

    length = len(string) + 1
    print_rank_0('-' * length)
    print_rank_0(string)
    print_rank_0('-' * length)


def build_train_valid_test_data_iterators(neox_args):
    """XXX"""

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0('> building train, validation, and test datasets ...')

    # Ensure only the first/last pipeline stages have data loaders
    if neox_args.is_pipe_parallel:
        is_first_stage = mpu.get_pipe_parallel_rank() == 0
        is_last_stage = mpu.get_pipe_parallel_rank() == mpu.get_pipe_parallel_world_size() - 1
        pipe_load = is_first_stage or is_last_stage
    else:
        pipe_load = True

    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_model_parallel_rank() == 0 and pipe_load:
        # Rank, size, and global batch size.
        data_parallel_size = mpu.get_data_parallel_world_size()
        global_batch_size = neox_args.batch_size * data_parallel_size * neox_args.gas

        # Number of train/valid/test samples.
        train_iters = neox_args.train_iters
        eval_iters = (train_iters // neox_args.eval_interval + 1) * neox_args.eval_iters
        test_iters = neox_args.eval_iters
        train_val_test_num_samples = [train_iters * global_batch_size,
                                      eval_iters * global_batch_size,
                                      test_iters * global_batch_size]
        print_rank_0(' > datasets target sizes (minimum size):')
        print_rank_0('    train:      {}'.format(train_val_test_num_samples[0]))
        print_rank_0('    validation: {}'.format(train_val_test_num_samples[1]))
        print_rank_0('    test:       {}'.format(train_val_test_num_samples[2]))

        # Build the datasets.

        print_rank_0('> building train, validation, and test datasets for GPT2 ...')
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            data_prefix=neox_args.data_path,
            data_impl=neox_args.data_impl,
            splits_string=neox_args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=neox_args.seq_length,
            seed=neox_args.seed,
            skip_warmup=(not neox_args.mmap_warmup)
            )
        print_rank_0("> finished creating GPT2 datasets ...")

        # Build dataloders.
        train_dataloader = make_data_loader(train_ds, neox_args=neox_args)
        valid_dataloader = make_data_loader(valid_ds, neox_args=neox_args)
        test_dataloader = make_data_loader(test_ds, neox_args=neox_args)

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and neox_args.train_iters > 0
        do_valid = valid_dataloader is not None and neox_args.eval_iters > 0
        do_test = test_dataloader is not None and neox_args.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        flags = torch.cuda.LongTensor(
            [int(do_train), int(do_valid), int(do_test)])
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    if neox_args.is_pipe_parallel:
        # Only first/last pipeline stages have data loaders, so pipeline parallelism should
        # broadcast globally instead of just the model parallel group.
        torch.distributed.broadcast(flags, src=0)
    else:
        torch.distributed.broadcast(flags,
                                    mpu.get_model_parallel_src_rank(),
                                    group=mpu.get_model_parallel_group())
    neox_args.do_train = flags[0].item()
    neox_args.do_valid = flags[1].item()
    neox_args.do_test = flags[2].item()

    # Shift the start iterations.
    if train_dataloader is not None:
        train_dataloader.batch_sampler.start_iter = neox_args.iteration % \
                                                    len(train_dataloader)
        print_rank_0('setting training data start iteration to {}'.
                     format(train_dataloader.batch_sampler.start_iter))
    if valid_dataloader is not None:
        start_iter_val = (neox_args.iteration // neox_args.eval_interval) * \
                         neox_args.eval_iters
        valid_dataloader.batch_sampler.start_iter = start_iter_val % \
                                                    len(valid_dataloader)
        print_rank_0('setting validation data start iteration to {}'.
                     format(valid_dataloader.batch_sampler.start_iter))

    # Build iterators.
    if train_dataloader is not None:
        train_data_iterator = iter(train_dataloader)
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = iter(valid_dataloader)
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = iter(test_dataloader)
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator


def get_total_params(model):
    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        params = sum([p.nelement() for p in model.parameters()])
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(), params), flush=True)
    else:
        params = 0

    total_n_parameters = torch.tensor([params]).cuda(torch.cuda.current_device())
    torch.distributed.all_reduce(total_n_parameters)
    total_n_parameters = total_n_parameters.item()
    return total_n_parameters


def human_readable_flops(num):
    for unit in ['', 'KFLOPS', 'MFLOPS', 'GFLOPS', 'TFLOPS', 'PFLOPS', 'EFLOPS', 'ZFLOPS']:
        if abs(num) < 1000.0:
            return "%3.1f%s" % (num, unit)
        num /= 1000.0
    return "%.1f%s" % (num, 'Yi')


def get_global_batch_size(neox_args):
    return neox_args.batch_size * mpu.get_data_parallel_world_size() * neox_args.gas


def get_flops(neox_args, model, iter_time_s):

    world_size = torch.distributed.get_world_size()
    global_batch_size = get_global_batch_size(neox_args)

    ff = model.total_params * 6
    attn = neox_args.seq_length * neox_args.hidden_size * neox_args.num_layers * 60
    flops = global_batch_size * neox_args.seq_length * (ff + attn) / (iter_time_s * world_size)

    return flops
