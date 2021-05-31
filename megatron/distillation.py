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

import sys

import torch

from megatron.utils import Timers, init_wandb
from megatron import print_rank_0

from megatron.checkpointing import load_checkpoint, save_checkpoint
from megatron.data.data_utils import build_train_valid_test_data_iterators

from megatron.initialize import initialize_megatron
from megatron.utils import OverflowMonitor, get_noise_scale_logger
from megatron.logging import training_log

from megatron.model.gpt2_model import cross_entropy, kldiv_loss, mse_loss
from megatron.utils import reduce_losses

from .training import setup_model_and_optimizer
from .training import evaluate_and_print_results
from .training import get_batch
from .training import backward_step


def distillation(neox_args):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the model.

    Arguments:
        neox_args: an instance of NeoXArgs containing the configuration for pretrain

    """

    if neox_args.is_pipe_parallel:
        raise NotImplementedError("Pipe parallel for distillation to be implemented!")

    # setup logging and timers
    init_wandb(neox_args=neox_args)
    timers = Timers(use_wandb=neox_args.use_wandb, tensorboard_writer=neox_args.tensorboard_writer)

    def substitue_args(neox_args, set_student_args=True):
        if neox_args.do_distillation:
            args_to_substitue = neox_args.student_model_args \
                if set_student_args else neox_args.teacher_model_args
            for arg in args_to_substitue.__dict__:
                if args_to_substitue.__dict__[arg] is not None:
                    neox_args.__dict__[arg] = args_to_substitue.__dict__[arg]
        return neox_args

    # Initalize and get arguments, timers, and Tensorboard writer.
    neox_args = substitue_args(neox_args)
    initialize_megatron(neox_args=neox_args)

    # Model, optimizer, and learning rate.
    timers('model and optimizer').start()

    neox_args = substitue_args(neox_args, set_student_args=False)
    neox_args.load = neox_args.load_teacher
    teacher_model, _, _ = setup_model_and_optimizer(neox_args=neox_args, inference=False, get_key_value=True)

    neox_args = substitue_args(neox_args, set_student_args=True)
    neox_args.load = neox_args.load_student
    student_model, optimizer, lr_scheduler = setup_model_and_optimizer(neox_args=neox_args, inference=False, get_key_value=True)

    timers('model and optimizer').stop()

    # Data stuff.
    timers('train/valid/test data iterators').start()
    train_data_iterator, valid_data_iterator, test_data_iterator = build_train_valid_test_data_iterators(
        neox_args=neox_args)
    timers('train/valid/test data iterators').stop()

    # Print setup timing.
    print_rank_0('done with setups ...')
    timers.log(['model and optimizer', 'train/valid/test data iterators'])
    print_rank_0('training ...')

    iteration = 0
    if neox_args.do_train and neox_args.train_iters > 0:
        iteration = distil(
            neox_args=neox_args,
            timers=timers,
            model=(teacher_model, student_model),
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
            model=student_model,
            iteration=iteration,
            verbose=False,
            timers=timers
        )

    if neox_args.save and iteration != 0:
        save_checkpoint(neox_args=neox_args, iteration=iteration, model=student_model, optimizer=optimizer,
                        lr_scheduler=lr_scheduler)

    if neox_args.do_test:
        # Run on test data.
        prefix = 'the end of training for test data'
        evaluate_and_print_results(
            neox_args=neox_args,
            prefix=prefix,
            forward_step_func=forward_step,
            data_iterator=test_data_iterator,
            model=student_model,
            iteration=0,  # iteration 0 in order to always use full test data
            verbose=True,
            timers=timers
        )

def forward_step(data_iterator, model, neox_args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(neox_args=neox_args,
                                                                        data_iterator=data_iterator)
    timers('batch generator').stop()

    if isinstance(model, tuple):
        teacher_model, student_model = model

        teacher_model.eval()
        with torch.no_grad():
            teacher_logits = teacher_model((tokens, position_ids, attention_mask))

        student_logits = student_model((tokens, position_ids, attention_mask))

        if neox_args.alpha_lm > 0:
            lm_loss = cross_entropy(student_logits, (labels, loss_mask), _fp16=neox_args.reduce_loss_fp16)
            loss = neox_args.alpha_lm * lm_loss

        if neox_args.alpha_kld > 0:
            kl_loss = kldiv_loss(student_logits, (teacher_logits, loss_mask), _fp16=neox_args.reduce_loss_fp16)
            loss += neox_args.alpha_kld * kl_loss

        if neox_args.alpha_mse > 0:
            ms_loss = mse_loss(student_logits, (teacher_logits, loss_mask), _fp16=neox_args.reduce_loss_fp16)
            loss += neox_args.alpha_mse * ms_loss
    else:
        outputs = model((tokens, position_ids, attention_mask))
        loss = cross_entropy(outputs, (labels, loss_mask), _fp16=neox_args.reduce_loss_fp16)

    return loss

def distil_step(neox_args, timers, data_iterator, model, optimizer, lr_scheduler):
    """Single training step."""

    # Pipeline parallelism schedules forward/backward/step
    teacher_model, student_model = model

    if neox_args.is_pipe_parallel:
        raise NotImplementedError("Pipe parallel for distillation to be implemented!")
        # reduced_loss = train_step_pipe(neox_args=neox_args, timers=timers, model=model, data_iterator=data_iterator)
    else:
        losses = []
        for _ in range(neox_args.gradient_accumulation_steps):
            # Forward model for one step.
            timers('forward').start()
            loss = forward_step(neox_args=neox_args, timers=timers, data_iterator=data_iterator, model=(teacher_model, student_model ))
            timers('forward').stop()
            losses.append(loss)
            # Calculate gradients, reduce across processes, and clip.
            timers('backward').start()
            backward_step(neox_args=neox_args, timers=timers, optimizer=optimizer, model=student_model, loss=loss)
            timers('backward').stop()
            # Update parameters.
            timers('optimizer').start()
            if neox_args.deepspeed:
                student_model.step()
            else:
                raise ValueError("Must be using deepspeed to run neox")
            timers('optimizer').stop()
        reduced_loss = {"lm_loss": reduce_losses(losses).mean()}  # reduces losses across machines for logging

    if neox_args.precision == "fp16" and student_model.optimizer.overflow:
        skipped_iter = 1
    else:
        skipped_iter = 0

    return reduced_loss, skipped_iter


def distil_step_pipe(neox_args, timers, model, data_iterator):
    """Single training step with DeepSpeed's pipeline parallel engine. """

    assert neox_args.deepspeed
    teacher_model, student_model = model
    loss = student_model.train_batch(data_iter=data_iterator)
    loss_dict = {'lm_loss': loss}
    # Don't break Megatron's timers because we changed code paths.
    for t in ['forward', 'backward', 'allreduce', 'optimizer', 'batch generator', 'data loader']:
        timers(t).reset()
    return loss_dict


def distil(neox_args, timers, model, optimizer, lr_scheduler,
          train_data_iterator, valid_data_iterator):
    """Train the model function."""

    # Turn on training mode which enables dropout.
    teacher_model, student_model = model
    teacher_model.eval()
    student_model.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = neox_args.iteration

    timers('interval time').start()
    report_memory_flag = True

    # get noise scale logger (if neox_args.log_gradient_noise_scale is True)
    noise_scale_logger = get_noise_scale_logger(neox_args)

    # to monitor if we've skipped many iterations in a row and trigger an early exit
    overflow_monitor = OverflowMonitor(optimizer)
    while iteration < neox_args.train_iters:
        loss_dict, skipped_iter = distil_step(
            neox_args=neox_args,
            timers=timers,
            data_iterator=train_data_iterator,
            model=(teacher_model, student_model),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler
        )

        iteration += 1

        overflow_monitor.check(skipped_iter)  # check for repeated overflow
        if neox_args.log_gradient_noise_scale:  # log noise scale if applicable
            noise_scale_logger.update()

        # Logging.
        report_memory_flag = training_log(
            neox_args=neox_args,
            timers=timers,
            loss_dict=loss_dict,
            total_loss_dict=total_loss_dict,
            learning_rate=optimizer.param_groups[0]['lr'],
            iteration=iteration,
            loss_scale=optimizer.cur_scale if neox_args.precision == "fp16" else None,
            report_memory_flag=report_memory_flag,
            skipped_iter=skipped_iter,
            model=student_model,
            optimizer=optimizer,
            noise_scale_logger=noise_scale_logger
        )

        # Checkpointing
        if neox_args.save and neox_args.save_interval and iteration % neox_args.save_interval == 0:
            save_checkpoint(neox_args=neox_args, iteration=iteration, model=student_model, optimizer=optimizer,
                            lr_scheduler=lr_scheduler)

        # Evaluation
        if neox_args.eval_interval and iteration % neox_args.eval_interval == 0 and neox_args.do_valid:
            prefix = 'iteration {}'.format(iteration)
            evaluate_and_print_results(
                neox_args=neox_args,
                prefix=prefix,
                forward_step_func=forward_step,
                data_iterator=valid_data_iterator,
                model=student_model,
                iteration=iteration,
                verbose=False,
                timers=timers
            )

        if neox_args.exit_interval and iteration % neox_args.exit_interval == 0:
            torch.distributed.barrier()
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            rank = torch.distributed.get_rank()
            print_rank_0('rank: {} | time: {} | exiting the program at iteration {}'.format(rank, time_str, iteration))
            sys.exit()

    return iteration
