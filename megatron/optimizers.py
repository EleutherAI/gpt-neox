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
from torch.optim import Optimizer


class SM3(Optimizer):
    """Implements SM3 algorithm.
    It has been proposed in `Memory-Efficient Adaptive Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): coefficient that scale delta before it is applied
            to the parameters (default: 0.1)
        momentum (float, optional): coefficient used to scale prior updates
            before adding. This drastically increases memory usage if
            `momentum > 0.0`. This is ignored if the parameter's gradient
            is sparse. (default: 0.0)
        beta (float, optional): coefficient used for exponential moving
            averages (default: 0.0)
        eps (float, optional): Term added to square-root in denominator to
            improve numerical stability (default: 1e-30)
    .. _Memory-Efficient Adaptive Optimization:
        https://arxiv.org/abs/1901.11150
    """

    def __init__(self, params, lr=0.1, momentum=0.0, beta=0.0, eps=1e-30):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {0}".format(lr))
        if not 0.0 <= momentum < 1.0:
            raise ValueError("Invalid momentum: {0}".format(momentum))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta: {0}".format(beta))
        if not 0.0 <= eps:
            raise ValueError("Invalid eps: {0}".format(eps))

        defaults = {"lr": lr, "momentum": momentum, "beta": beta, "eps": eps}
        super(SM3, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            beta = group["beta"]
            eps = group["eps"]
            for p in group["params"]:
                if p is None:
                    continue
                grad = p.grad

                state = self.state[p]
                shape = grad.shape
                rank = len(shape)

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = 0.0
                    _add_initial_accumulators(state, grad)

                if grad.is_sparse:
                    # the update is non-linear so indices must be unique
                    grad.coalesce()
                    grad_indices = grad._indices()
                    grad_values = grad._values()

                    # Transform update_values into sparse tensor
                    def make_sparse(values):
                        constructor = grad.new
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor().resize_as_(grad)
                        return constructor(grad_indices, values, grad.size())

                    acc = state[_key(0)]
                    update_values = _compute_sparse_update(
                        beta, acc, grad_values, grad_indices
                    )

                    self._update_sparse_accumulator(
                        beta, acc, make_sparse(update_values)
                    )

                    # Add small amount for numerical stability
                    update_values.add_(eps).rsqrt_().mul_(grad_values)

                    update = make_sparse(update_values)
                else:
                    # Get previous accumulators mu_{t-1}
                    if rank > 1:
                        acc_list = [state[_key(i)] for i in range(rank)]
                    else:
                        acc_list = [state[_key(0)]]

                    # Get update from accumulators and gradients
                    update = _compute_update(beta, acc_list, grad)

                    # Update accumulators.
                    self._update_accumulator(beta, acc_list, update)

                    # Add small amount for numerical stability
                    update.add_(eps).rsqrt_().mul_(grad)

                    if momentum > 0.0:
                        m = state["momentum_buffer"]
                        update.mul_(1.0 - momentum).add_(m, alpha=momentum)
                        state["momentum_buffer"] = update.detach()

                p.sub_(update, alpha=group["lr"])
                state["step"] += 1
        return loss

    @staticmethod
    def _update_accumulator(beta, acc_list, update):
        for i, acc in enumerate(acc_list):
            nu_max = _max_reduce_except_dim(update, i)
            if beta > 0.0:
                torch.max(acc, nu_max, out=acc)
            else:
                # No need to compare - nu_max is bigger because of grad ** 2
                acc.copy_(nu_max)

    @staticmethod
    def _update_sparse_accumulator(beta, acc, update):
        nu_max = _max_reduce_except_dim(update.to_dense(), 0).squeeze()
        if beta > 0.0:
            torch.max(acc, nu_max, out=acc)
        else:
            # No need to compare - nu_max is bigger because of grad ** 2
            acc.copy_(nu_max)


def _compute_sparse_update(beta, acc, grad_values, grad_indices):
    # In the sparse case, a single accumulator is used.
    update_values = torch.gather(acc, 0, grad_indices[0])
    if beta > 0.0:
        update_values.mul_(beta)
    update_values.addcmul_(grad_values, grad_values, value=1.0 - beta)
    return update_values


def _compute_update(beta, acc_list, grad):
    rank = len(acc_list)
    update = acc_list[0].clone()
    for i in range(1, rank):
        # We rely on broadcasting to get the proper end shape.
        update = torch.min(update, acc_list[i])
    if beta > 0.0:
        update.mul_(beta)
    update.addcmul_(grad, grad, value=1.0 - beta)

    return update


def _key(i):
    # Returns key used for accessing accumulators
    return "accumulator_" + str(i)


def _add_initial_accumulators(state, grad):
    # Creates initial accumulators. For a dense tensor of shape (n1, n2, n3),
    # then our initial accumulators are of shape (n1, 1, 1), (1, n2, 1) and
    # (1, 1, n3). For a sparse tensor of shape (n, *), we use a single
    # accumulator of shape (n,).
    shape = grad.shape
    rank = len(shape)
    defaults = {"device": grad.device, "dtype": grad.dtype}
    acc = {}

    if grad.is_sparse:
        acc[_key(0)] = torch.zeros(shape[0], **defaults)
    elif rank == 0:
        # The scalar case is handled separately
        acc[_key(0)] = torch.zeros(shape, **defaults)
    else:
        for i in range(rank):
            acc_shape = [1] * i + [shape[i]] + [1] * (rank - 1 - i)
            acc[_key(i)] = torch.zeros(acc_shape, **defaults)

    state.update(acc)


def _max_reduce_except_dim(tensor, dim):
    # Computes max along all dimensions except the given dim.
    # If tensor is a scalar, it returns tensor.
    rank = len(tensor.shape)
    result = tensor
    if rank > 0:
        assert dim < rank
        for d in range(rank):
            if d != dim:
                result = result.max(dim=d, keepdim=True).values
    return result


# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# modifications  - 4/4/2021  @lessw2020  (decay issue spotted by @nestordemeure )
# weight decay has been implemented AdamW style instead of the original madgrad Adam style.
# in initial image classification testing, this outperformed 0 weight decay or original style weight decay.

# closure is checked if callable or not since some code passes loss directly, rather than in closure param

import math
from typing import Collection, TYPE_CHECKING, Any, Callable, Optional, Tuple

import torch
import torch.optim
import collections

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any


class madgrad_wd(torch.optim.Optimizer):
    """
    MADGRAD_: A Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic
    Optimization.

    .. _MADGRAD: https://arxiv.org/abs/2101.11075

    MADGRAD is a general purpose optimizer that can be used in place of SGD or
    Adam may converge faster and generalize better. Currently GPU-only.
    Typically, the same learning rate schedule that is used for SGD or Adam may
    be used. The overall learning rate is not comparable to either method and
    should be determined by a hyper-parameter sweep.

    MADGRAD requires less weight decay than other methods, often as little as
    zero. Momentum values used for SGD or Adam's beta1 should work here also.

    On sparse problems both weight_decay and momentum should be set to 0.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float):
            Learning rate (default: 1e-2).
        momentum (float):
            Momentum value in  the range [0,1) (default: 0.9).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        eps (float):
            Term added to the denominator outside of the root operation to improve numerical stability. (default: 1e-6).
    """

    def __init__(
        self,
        params: _params_t,
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 0,
        eps: float = 1e-6,
    ):
        if momentum < 0 or momentum >= 1:
            raise ValueError(f"Momentum {momentum} must be in the range [0,1]")
        if lr <= 0:
            raise ValueError(f"Learning rate {lr} must be positive")
        if weight_decay < 0:
            raise ValueError(f"Weight decay {weight_decay} must be non-negative")
        if eps < 0:
            raise ValueError(f"Eps must be non-negative")

        defaults = dict(lr=lr, eps=eps, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self) -> bool:
        return False

    @property
    def supports_flat_params(self) -> bool:
        return True

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None and isinstance(closure, collections.Callable):
            loss = closure()

        # step counter must be stored in state to ensure correct behavior under
        # optimizer sharding
        if "k" not in self.state:
            self.state["k"] = torch.tensor([0], dtype=torch.long)
        k = self.state["k"].item()

        for group in self.param_groups:
            eps = group["eps"]
            lr = group["lr"] + eps
            decay = group["weight_decay"]
            momentum = group["momentum"]

            ck = 1 - momentum
            lamb = lr * math.pow(k + 1, 0.5)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if "grad_sum_sq" not in state:
                    state["grad_sum_sq"] = torch.zeros_like(p.data).detach()
                    state["s"] = torch.zeros_like(p.data).detach()
                    if momentum != 0:
                        state["x0"] = torch.clone(p.data).detach()

                if momentum != 0.0 and grad.is_sparse:
                    raise RuntimeError(
                        "momentum != 0 is not compatible with sparse gradients"
                    )

                grad_sum_sq = state["grad_sum_sq"]
                s = state["s"]

                # Apply weight decay - L2 / AdamW style
                if decay:
                    p.data.mul_(1 - lr * decay)

                """ original impl:
                if decay != 0:
                    if grad.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients")

                    grad.add_(p.data, alpha=decay)
                """

                if grad.is_sparse:
                    grad = grad.coalesce()
                    grad_val = grad._values()

                    p_masked = p.sparse_mask(grad)
                    grad_sum_sq_masked = grad_sum_sq.sparse_mask(grad)
                    s_masked = s.sparse_mask(grad)

                    # Compute x_0 from other known quantities
                    rms_masked_vals = grad_sum_sq_masked._values().pow(1 / 3).add_(eps)
                    x0_masked_vals = p_masked._values().addcdiv(
                        s_masked._values(), rms_masked_vals, value=1
                    )

                    # Dense + sparse op
                    grad_sq = grad * grad
                    grad_sum_sq.add_(grad_sq, alpha=lamb)
                    grad_sum_sq_masked.add_(grad_sq, alpha=lamb)

                    rms_masked_vals = grad_sum_sq_masked._values().pow_(1 / 3).add_(eps)

                    s.add_(grad, alpha=lamb)
                    s_masked._values().add_(grad_val, alpha=lamb)

                    # update masked copy of p
                    p_kp1_masked_vals = x0_masked_vals.addcdiv(
                        s_masked._values(), rms_masked_vals, value=-1
                    )
                    # Copy updated masked p to dense p using an add operation
                    p_masked._values().add_(p_kp1_masked_vals, alpha=-1)
                    p.data.add_(p_masked, alpha=-1)
                else:
                    if momentum == 0:
                        # Compute x_0 from other known quantities
                        rms = grad_sum_sq.pow(1 / 3).add_(eps)
                        x0 = p.data.addcdiv(s, rms, value=1)
                    else:
                        x0 = state["x0"]

                    # Accumulate second moments
                    grad_sum_sq.addcmul_(grad, grad, value=lamb)
                    rms = grad_sum_sq.pow(1 / 3).add_(eps)

                    # Update s
                    s.data.add_(grad, alpha=lamb)

                    # Step
                    if momentum == 0:
                        p.data.copy_(x0.addcdiv(s, rms, value=-1))
                    else:
                        z = x0.addcdiv(s, rms, value=-1)

                        # p is a moving average of z
                        p.data.mul_(1 - ck).add_(z, alpha=ck)

        self.state["k"] += 1
        return loss


class Lion(Optimizer):
    """
    Implements the Lion Algorithm

    .. / _Lion: https://arxiv.org/abs/2302.06675

    Compared to AdamW and various adaptive optimizers that need to save both first and second moments,
    Lion only needs the momentum, halving the additional memory footprint. This is beneficial when training large models
    and / or with a large batch size.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float):
            Learning rate (default: 1e-2).
        beta (float):
            coefficients used for computing running averages of gradient and its square (default: (0.9, 0.99))
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).

    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        if lr <= 0:
            raise ValueError(f"Learning rate {lr} must be positive")
        if weight_decay < 0:
            raise ValueError(f"Weight decay {weight_decay} must be non-negative")
        if not (0 <= betas[0] <= 1 and 0 <= betas[1] <= 1):
            raise ValueError(f"Betas {betas} must be in range [0, 1)")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def update(self, p, grad, exp_avg, lr, wd, beta1, beta2):
        """https://arxiv.org/pdf/2302.06675.pdf#appendix.A"""

        # update model parameters
        p.mul_(1 - lr * wd)
        sign = exp_avg.clone().mul_(beta1).add(grad, alpha=1 - beta1).sign_()
        p.add_(sign, alpha=-lr)

        # update EMA
        exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                # init state - exponential moving average of gradient values
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p.data).detach()

                self.update(
                    p,
                    p.grad,
                    state["exp_avg"],
                    group["lr"],
                    group["weight_decay"],
                    group["betas"][0],
                    group["betas"][1],
                )

        return loss
