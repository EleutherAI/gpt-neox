# Copyright (c) 2024, EleutherAI
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

import torch


def ema(avg, beta, yi, i):
    """Exponential moving average"""
    if avg is None:
        avg = 0
    avg = beta * avg + (1 - beta) * yi
    return avg, avg / (1 - beta ** (i + 1))


class GradientNoiseScale:
    """
    A class to measure the gradient noise scale of a model while training (cf. https://arxiv.org/abs/1812.06162).

    The core thesis of the paper is that, if our batch size is small, there will be a lot of noise present in the gradients, and we might update our weights only on noise.
    After several updates the optimizer may still push us in the right direction, but we would be better off having used a larger batch size, which is more computationally
    efficient and directly averages out the noise in the gradients.

    But there's a limit to the gains large batch sizes can give you - if, after a certain batch size, your gradient is already accurate, there's no point in increasing the
    batch size further, as we'll just be wasting compute for little to no gain in accuracy.

    This means there is some theoretically optimal batch size for a given model, which measuring the gradient noise scale can help us to estimate.

    To estimate the 'simple' noise scale (Bsimple), we need to have a measure of the gradients using a large batch size (Bbig) and a small
    batch size (Bsmall).

    when we have those:
        Bsimple ≈ (tr(Σ) / |G|^2)

    tr(Σ) can be approximated by:
        tr(Σ) ≈ (1 / ((1/Bsmall) - (1/Bbig))) * (|Gsmall|^2 - |Gbig|^2)

    and |G|^2 by:
        |G|^2 ≈ (1 / (Bbig - Bsmall)) * (Bbig*|Gbig|^2 - Bsmall*|Gsmall|^2)

    - With multi-gpu training, we can do this by taking the gradients of the microbatch_size_per_gpu for Bsmall,
    and the gradients of the entire batch for Bbig.
    - Alternatively, we can just take Bsmall as a single batch, and Bbig as several sequential batches in a row.
    This is the option we've opted for in this implementation because a) it's easier to implement and b) also works in
    single-gpu environments. Unfortunately it does come with some memory overhead.
    """

    def __init__(
        self,
        model,
        batch_size_small,
        n_batches=10,
        beta=0.99,
        cpu_offload=False,
        neox_args=None,
        mpu=None,
    ):
        self.batch_size_small = batch_size_small
        self.batch_size_large = batch_size_small * n_batches
        self.n_batches = n_batches
        self.beta = beta
        self.model = model
        self.buffer = None
        self.ema_scale = None
        self.ema_noise = None
        self.noise_scale = None
        self.n_updates = 0
        self.cpu_offload = cpu_offload
        self.model.store_gradients = True
        self.model.store_gradients_cpu = cpu_offload
        self.neox_args = neox_args
        self.mpu = mpu

    def flatten_grads(self):
        grads = []
        assert hasattr(
            self.model, "stored_gradients"
        ), "You might need to update DeeperSpeed"
        if self.model.stored_gradients is not None:
            for g in self.model.stored_gradients:
                if g is not None and not g.isnan().any() and not g.isinf().any():
                    g = g.flatten().view(-1, 1)
                    if self.cpu_offload:
                        g = g.cpu()
                    grads.append(g)
                else:
                    return None
            if not grads:
                return None
            return torch.cat(grads)

    def _sync_overflow(self, is_overflow):
        if self.neox_args.is_pipe_parallel:
            # Since each model parallel GPU carries only part of the model,
            # make sure overflow flag is synced across all the pipe parallel GPUs
            overflow_gpu = torch.cuda.ByteTensor([is_overflow])
            torch.distributed.all_reduce(
                overflow_gpu,
                op=torch.distributed.ReduceOp.MAX,
                group=self.mpu.get_pipe_parallel_group(),
            )
            overflow = overflow_gpu[0].item()
        else:
            overflow = is_overflow
        return overflow

    def _update(self):

        grad = self.flatten_grads()
        is_overflow = self._sync_overflow(grad is None)
        if is_overflow:
            return
        if self.buffer is None:
            self.buffer = grad
        else:
            self.buffer += grad
        if self.n_updates % self.n_batches == self.n_batches - 1:
            # average grads every n_batches iteration to get a simulation of Bbig
            self.buffer /= self.n_batches
            grads = self.buffer
            self.buffer = None

            # calculate Gbig and Gsmall
            # this needs to be done in fp32 or it overflows
            if self.neox_args.is_pipe_parallel:

                g_big = torch.square(torch.norm(grads.to(torch.float)))
                g_small = torch.square(torch.norm(grad.to(torch.float)))

                # we need to put the tensors back on gpu to do the allreduce
                if self.cpu_offload:
                    g_big = g_big.to(self.model.device)
                    g_small = g_small.to(self.model.device)

                # avg g_big / g_small across pipe parallel groups
                torch.distributed.all_reduce(
                    g_big,
                    op=torch.distributed.ReduceOp.SUM,
                    group=self.mpu.get_pipe_parallel_group(),
                )
                torch.distributed.all_reduce(
                    g_small,
                    op=torch.distributed.ReduceOp.SUM,
                    group=self.mpu.get_pipe_parallel_group(),
                )
                g_big /= self.mpu.get_pipe_parallel_world_size()
                g_small /= self.mpu.get_pipe_parallel_world_size()

            else:
                g_big = torch.square(torch.norm(grads.to(torch.float)))
                g_small = torch.square(torch.norm(grad.to(torch.float)))

            # communicate any overflows
            is_overflow = (
                g_small.isinf().any()
                or g_small.isnan().any()
                or g_big.isinf().any()
                or g_big.isnan().any()
            )
            is_overflow = self._sync_overflow(is_overflow)
            if is_overflow:
                return

            # calculate noise / scale
            noise = (
                1
                / (self.batch_size_large - self.batch_size_small)
                * (self.batch_size_large * g_big - self.batch_size_small * g_small)
            )
            scale = (
                1
                / (1 / self.batch_size_small - 1 / self.batch_size_large)
                * (g_small - g_big)
            )

            # calculate running average
            self.ema_noise, noise = ema(
                self.ema_noise, self.beta, noise, self.n_updates
            )
            self.ema_scale, scale = ema(
                self.ema_scale, self.beta, scale, self.n_updates
            )

            # calculate noise scale
            scale = scale.item()
            noise = noise.item()
            self.noise_scale = scale / noise

        self.n_updates += 1

    def update(self):
        if self.neox_args.is_pipe_parallel:
            # update on all ranks
            self._update()
        else:
            # for mp / dp only, the grads will be the same across all ranks, so we can just do the process on a single rank
            if torch.distributed.get_rank() == 0:
                # only update on 0th rank
                self._update()
            torch.distributed.barrier()
