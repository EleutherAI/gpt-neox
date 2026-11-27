# Copyright (c) 2025, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


from typing import Optional, Tuple

import torch

from .initialize import get_model_parallel_group
from .initialize import get_model_parallel_rank
from .initialize import get_model_parallel_world_size
from .utils import VocabUtility


def calculate_loss(
    predicted_logits: torch.Tensor,
    sum_exp_logits: torch.Tensor,
    logits_max: torch.Tensor,
    compute_z_loss: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Per-token CE, raw Z-loss, and its derivative with respect to log Z."""
    log_sum_exp = sum_exp_logits.log()
    loss = log_sum_exp - predicted_logits
    if not compute_z_loss:
        return loss, None, None

    # The target logit is max-shifted for CE. Z-loss needs the original log Z.
    log_z = log_sum_exp + logits_max
    return loss, log_z.square(), 2.0 * log_z


def calculate_gradients(
    softmax: torch.Tensor,
    grad_output: torch.Tensor,
    target_mask: torch.Tensor,
    masked_target_1d: torch.Tensor,
    logits_dtype: torch.dtype,
    z_loss_grad: Optional[torch.Tensor],
    grad_penalty: Optional[torch.Tensor],
) -> torch.Tensor:
    """Reuse the saved softmax as the gradient buffer for this vocabulary shard."""
    grad_2d = softmax.view(-1, softmax.size(-1))
    rows = torch.arange(grad_2d.size(0), device=softmax.device)
    local_target = 1.0 - target_mask.reshape(-1).float()
    if z_loss_grad is None or grad_penalty is None:
        grad_2d[rows, masked_target_1d] -= local_target
        softmax.mul_(grad_output.unsqueeze(-1))
    else:
        # Subtract the CE target first so a small Z gradient survives when p(target) rounds to 1.
        target_probs = grad_2d[rows, masked_target_1d]
        grad_log_z = grad_penalty * z_loss_grad
        target_grad = torch.addcmul(
            (target_probs - local_target) * grad_output.reshape(-1),
            target_probs,
            grad_log_z.reshape(-1),
        )
        softmax.mul_((grad_output + grad_log_z).unsqueeze(-1))
        grad_2d[rows, masked_target_1d] = target_grad
    return softmax.to(logits_dtype)


class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, compute_z_loss):
        ctx.logits_dtype = vocab_parallel_logits.dtype
        if compute_z_loss and ctx.logits_dtype in (torch.float16, torch.bfloat16):
            # The joint loss shares the FP32 normalization required by Z-loss.
            vocab_parallel_logits = vocab_parallel_logits.float()

        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        torch.distributed.all_reduce(
            logits_max,
            op=torch.distributed.ReduceOp.MAX,
            group=get_model_parallel_group(),
        )
        # Subtract the maximum value.
        vocab_parallel_logits.sub_(logits_max.unsqueeze(dim=-1))

        # Get the partition's vocab indices
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        rank = get_model_parallel_rank()
        world_size = get_model_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(
            partition_vocab_size, rank, world_size
        )

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(
            start=0, end=logits_2d.size()[0], device=logits_2d.device
        )
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(
            predicted_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_model_parallel_group(),
        )

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = vocab_parallel_logits
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        torch.distributed.all_reduce(
            sum_exp_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_model_parallel_group(),
        )

        loss, penalty, z_loss_grad = calculate_loss(
            predicted_logits, sum_exp_logits, logits_max, compute_z_loss
        )

        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d, z_loss_grad)

        return loss, penalty

    @staticmethod
    def backward(ctx, grad_output, grad_penalty):
        softmax, target_mask, masked_target_1d, z_loss_grad = ctx.saved_tensors
        grad_input = calculate_gradients(
            softmax,
            grad_output,
            target_mask,
            masked_target_1d,
            ctx.logits_dtype,
            z_loss_grad,
            grad_penalty,
        )
        return grad_input, None, None


def vocab_parallel_cross_entropy(
    vocab_parallel_logits, target, z_loss=0.0, return_components=False
):
    """Return the per-token objective or the separate CE and raw Z-loss."""
    loss, penalty = _VocabParallelCrossEntropy.apply(
        vocab_parallel_logits, target, z_loss != 0.0
    )
    if return_components:
        return loss, penalty
    return loss if penalty is None else loss + z_loss * penalty
