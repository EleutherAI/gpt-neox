# Copyright (c) 2021, EleutherAI.
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

import sys

import torch

try:
    import wandb
except ModuleNotFoundError:
    pass

from megatron import mpu, print_rank_0
from megatron.utils import report_memory


class Tee:
    """Duplicate output to both stdout/err and file"""

    def __init__(self, file, err: bool = False) -> None:
        self.file = open(file, "w")
        self.err = err
        if not err:
            self.std = sys.stdout
            sys.stdout = self
        else:
            self.std = sys.stderr
            sys.stderr = self

    def __del__(self) -> None:
        if not self.err:
            sys.stdout = self.std
        else:
            sys.stderr = self.std
        self.file.close()

    def write(self, data) -> None:
        try:
            self.file.write(data)
        except OSError:
            pass
        try:
            self.std.write(data)
        except OSError:
            pass

    def flush(self) -> None:
        try:
            self.file.flush()
        except OSError:
            pass


def human_readable_flops(num) -> str:
    for unit in [
        "",
        "KFLOPS",
        "MFLOPS",
        "GFLOPS",
        "TFLOPS",
        "PFLOPS",
        "EFLOPS",
        "ZFLOPS",
    ]:
        if abs(num) < 1000.0:
            return "%3.1f%s" % (num, unit)
        num /= 1000.0
    return "%.1f%s" % (num, "Yi")


def get_flops(neox_args, iter_time_s) -> float:
    """
    Use FLOPS calculation from Megatron-DeepSpeed:
    https://github.com/microsoft/Megatron-DeepSpeed/blob/cc3a94c636789f74be2bc6cfc62a3d723fd5d749/megatron/utils.py#L253
    They get it from https://arxiv.org/pdf/2104.04473.pdf
    """
    world_size = torch.distributed.get_world_size()
    vocab_size = neox_args.padded_vocab_size
    batch_size = neox_args.train_batch_size
    seq_len = neox_args.seq_length
    hidden_size = neox_args.hidden_size
    num_layers = neox_args.num_layers
    ckpt_activations_factor = 4 if neox_args.checkpoint_activations else 3
    flops_calc1 = (
        24
        * ckpt_activations_factor
        * batch_size
        * seq_len
        * num_layers
        * (hidden_size**2)
        * (1.0 + (seq_len / (6.0 * hidden_size)))
    )
    flops_calc2 = vocab_size / (16.0 * num_layers * hidden_size)
    flops_per_iteration = flops_calc1 + flops_calc2
    return flops_per_iteration / (iter_time_s * world_size)


def training_log(
    neox_args,
    timers,
    loss_dict,
    total_loss_dict,
    learning_rate,
    iteration,
    loss_scale,
    report_memory_flag,
    skipped_iter,
    model,
    optimizer,
    noise_scale_logger,
):
    """Log training information such as losses, timing, etc."""

    # Update losses.
    skipped_iters_key = "skipped iterations"
    total_loss_dict[skipped_iters_key] = (
        total_loss_dict.get(skipped_iters_key, 0) + skipped_iter
    )
    got_nan_key = "got nan"

    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = total_loss_dict.get(key, 0.0) + loss_dict[key]
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float("inf") or value == -float("inf") or value != value
            got_nan = got_nan or is_nan

    total_loss_dict[got_nan_key] = total_loss_dict.get(got_nan_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = []

    def add_to_logging(name):
        if name in timers.timers:
            timers_to_log.append(name)

    if not neox_args.is_pipe_parallel:
        add_to_logging("forward")
        add_to_logging("backward")
        add_to_logging("backward-backward")
        add_to_logging("backward-allreduce")
        add_to_logging("backward-master-grad")
        add_to_logging("backward-clip-grad")
        add_to_logging("optimizer")
        add_to_logging("batch generator")

        # Log timer info to tensorboard and wandb
        normalizer = iteration % neox_args.log_interval
        if normalizer == 0:
            normalizer = neox_args.log_interval
        if torch.distributed.get_rank() == 0:
            timers.write(
                names=timers_to_log, iteration=iteration, normalizer=normalizer
            )
    else:
        # with pipeline parallel, the megatron timers are overridden by the deepspeed ones.
        # Try to grab timer values from model engine. Only recently added to deeperspeed, so check that the engine
        # has that attribute first
        if hasattr(model, "timer_values") and model.timer_values is not None:
            if (
                model.wall_clock_breakdown()
                and model.global_steps % model.steps_per_print() == 0
            ):
                timer_values = model.timer_values
                # deepspeed already logs to tensorboard / prints values, so just log to wandb
                if neox_args.use_wandb and torch.distributed.get_rank() == 0:
                    for key in timer_values:
                        tb_wandb_log(
                            f"timers/{key}",
                            timer_values[key],
                            iteration,
                            use_wandb=neox_args.use_wandb,
                            tensorboard_writer=neox_args.tensorboard_writer,
                        )

    # write losses, lr, etc. every step
    tb_wandb_log(
        "train/learning_rate",
        learning_rate,
        iteration,
        use_wandb=neox_args.use_wandb,
        tensorboard_writer=neox_args.tensorboard_writer,
    )
    for key in loss_dict:
        tb_wandb_log(
            f'train/{key.replace(" ", "_")}',
            loss_dict[key],
            iteration,
            use_wandb=neox_args.use_wandb,
            tensorboard_writer=neox_args.tensorboard_writer,
        )
    if neox_args.fp16:
        tb_wandb_log(
            f"train/loss_scale",
            loss_scale,
            iteration,
            use_wandb=neox_args.use_wandb,
            tensorboard_writer=neox_args.tensorboard_writer,
        )

    # log gradient noise scale
    if neox_args.log_gradient_noise_scale:
        if noise_scale_logger.noise_scale is not None:
            tb_wandb_log(
                f"train/noise_scale",
                noise_scale_logger.noise_scale,
                iteration,
                use_wandb=neox_args.use_wandb,
                tensorboard_writer=neox_args.tensorboard_writer,
            )

    # (optional) Log optimizer states to wandb / tb every step
    if neox_args.log_optimizer_states:
        for k, v in optimizer.state_dict()["optimizer_state_dict"]["state"].items():
            for ki, vi in v.items():  # step, module
                if ki != "step":
                    opt_state_norm = torch.norm(vi) if hasattr(vi, "dim") else vi
                    tb_wandb_log(
                        f"optimizer_state_norms/{k}_{ki}",
                        opt_state_norm,
                        iteration,
                        use_wandb=neox_args.use_wandb,
                        tensorboard_writer=neox_args.tensorboard_writer,
                    )

    # (optional) Log grad/param norms to wandb / tb every step
    if (
        neox_args.log_grad_pct_zeros
        or neox_args.log_grad_norm
        or neox_args.log_param_norm
    ):
        if neox_args.log_grad_pct_zeros or neox_args.log_grad_norm:
            model.store_gradients = True  # start storing gradients

        for i, (name, param) in enumerate(model.module.named_parameters()):
            if neox_args.log_grad_pct_zeros:
                if (
                    hasattr(model, "stored_gradients")
                    and model.stored_gradients is not None
                ):
                    grad = model.stored_gradients[i]
                    if grad is not None:
                        tb_wandb_log(
                            f"pct_grad_zeros/{name}",
                            (grad == 0).float().mean().item() * 100,
                            iteration,
                            use_wandb=neox_args.use_wandb,
                            tensorboard_writer=neox_args.tensorboard_writer,
                            all_ranks=True,
                        )
            if neox_args.log_grad_norm:
                if (
                    hasattr(model, "stored_gradients")
                    and model.stored_gradients is not None
                ):
                    grad = model.stored_gradients[i]
                    if grad is not None:
                        tb_wandb_log(
                            f"gradient_norms/{name}",
                            torch.norm(grad),
                            iteration,
                            use_wandb=neox_args.use_wandb,
                            tensorboard_writer=neox_args.tensorboard_writer,
                            all_ranks=True,
                        )
            if neox_args.log_param_norm:
                tb_wandb_log(
                    f"parameter_norms/{name}",
                    torch.norm(param),
                    iteration,
                    use_wandb=neox_args.use_wandb,
                    tensorboard_writer=neox_args.tensorboard_writer,
                    all_ranks=True,
                )

    if iteration % neox_args.log_interval == 0:
        # log other stuff every neox_args.log_interval iters
        elapsed_time = timers("interval time").elapsed()
        iteration_time = elapsed_time / neox_args.log_interval
        samples_per_sec = neox_args.train_batch_size / iteration_time
        log_string = " samples/sec: {:.3f} |".format(samples_per_sec)
        tb_wandb_log(
            "runtime/samples_per_sec",
            samples_per_sec,
            iteration,
            use_wandb=neox_args.use_wandb,
            tensorboard_writer=neox_args.tensorboard_writer,
        )
        tb_wandb_log(
            "runtime/iteration_time",
            iteration_time,
            iteration,
            use_wandb=neox_args.use_wandb,
            tensorboard_writer=neox_args.tensorboard_writer,
        )
        log_string += " iteration {:8d}/{:8d} |".format(
            iteration, neox_args.train_iters
        )
        log_string += " elapsed time per iteration (ms): {:.1f} |".format(
            elapsed_time * 1000.0 / neox_args.log_interval
        )
        log_string += " learning rate: {:.3E} |".format(learning_rate)
        num_iterations = max(
            1, neox_args.log_interval - total_loss_dict[skipped_iters_key]
        )

        # log curriculum learning
        if neox_args.curriculum_learning:
            tb_wandb_log(
                "curriculum_seqlen",
                neox_args.curriculum_seqlen,
                iteration,
                use_wandb=neox_args.use_wandb,
                tensorboard_writer=neox_args.tensorboard_writer,
            )

        # log tflop / gpu
        flops_per_s_per_gpu = get_flops(neox_args, iteration_time)

        log_string += (
            f" approx flops per GPU: {human_readable_flops(flops_per_s_per_gpu)} |"
        )
        tb_wandb_log(
            "runtime/flops_per_sec_per_gpu",
            flops_per_s_per_gpu,
            iteration,
            use_wandb=neox_args.use_wandb,
            tensorboard_writer=neox_args.tensorboard_writer,
        )

        for key in total_loss_dict:
            if key not in [skipped_iters_key, got_nan_key]:
                v = (
                    total_loss_dict[key].item()
                    if hasattr(total_loss_dict[key], "item")
                    else total_loss_dict[key]
                )
                avg = v / float(num_iterations)
                log_string += " {}: {:.6E} |".format(key, avg)
                total_loss_dict[key] = 0.0
        if neox_args.precision == "fp16":
            log_string += " loss scale: {:.1f} |".format(loss_scale)
        log_string += " number of skipped iterations: {:3d} |".format(
            total_loss_dict[skipped_iters_key]
        )
        log_string += " number of nan iterations: {:3d} |".format(
            total_loss_dict[got_nan_key]
        )
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[got_nan_key] = 0
        print_rank_0(log_string)
        if report_memory_flag:
            report_memory("after {} iterations".format(iteration))
            report_memory_flag = False

        timers.log(timers_to_log, normalizer=neox_args.log_interval)

    return report_memory_flag


def tb_wandb_log(
    key: str,
    value: float,
    iteration_no: int,
    use_wandb: bool,
    tensorboard_writer=None,
    all_ranks: bool = False,
):
    # logs to both tb and wandb (if present) from the zeroth rank
    do_log = torch.distributed.get_rank() == 0 or all_ranks
    if do_log and value is not None:
        if tensorboard_writer:
            tensorboard_writer.add_scalar(key, value, iteration_no)
        if use_wandb:
            wandb.log({key: value}, step=iteration_no)
