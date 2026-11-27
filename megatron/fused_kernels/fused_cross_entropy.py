# Copyright (c) 2026, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

"""Fused vocabulary-parallel cross entropy.

This implementation combines the target-logit and exponential-sum payloads so
that tensor-parallel execution needs two all-reduces instead of three.
"""

from typing import Optional, Tuple

import torch

from megatron.mpu.cross_entropy import calculate_gradients, calculate_loss
from megatron.mpu.utils import VocabUtility


@torch.jit.script
def calculate_logits_max(
    vocab_parallel_logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Upcast logits and calculate each token's local maximum."""
    vocab_parallel_logits = vocab_parallel_logits.float()
    logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
    return vocab_parallel_logits, logits_max


@torch.jit.script
def calculate_predicted_logits(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    logits_max: torch.Tensor,
    vocab_start_index: int,
    vocab_end_index: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate local target logits and exponential sums as one payload."""
    vocab_parallel_logits -= logits_max.unsqueeze(dim=-1)

    target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
    masked_target = target.clone() - vocab_start_index
    masked_target[target_mask] = 0

    partition_vocab_size = vocab_parallel_logits.size(-1)
    logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
    masked_target_1d = masked_target.view(-1)
    arange_1d = torch.arange(logits_2d.size(0), device=logits_2d.device)
    predicted_logits = logits_2d[arange_1d, masked_target_1d].clone()
    predicted_logits = predicted_logits.view_as(target)
    predicted_logits[target_mask] = 0.0

    # Inplace exp to avoid an extra allocation and copy.
    exp_logits = vocab_parallel_logits
    torch.exp(vocab_parallel_logits, out=exp_logits)
    sum_exp_logits = exp_logits.sum(dim=-1)
    # Concatenate the predicted logits and the sum of exponentials so that we can do a single all-reduce.
    predicted_logits_sum_exp_logits = torch.cat(
        (predicted_logits, sum_exp_logits), dim=0
    )

    return (
        target_mask,
        masked_target_1d,
        predicted_logits_sum_exp_logits,
        exp_logits,
    )


@torch.jit.script
def calculate_cross_entropy_loss(
    exp_logits: torch.Tensor,
    predicted_logits_sum_exp_logits: torch.Tensor,
    logits_max: torch.Tensor,
    compute_z_loss: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Normalize probabilities and calculate the per-token loss."""
    split_at = predicted_logits_sum_exp_logits.size(0) // 2
    predicted_logits = predicted_logits_sum_exp_logits[:split_at]
    sum_exp_logits = predicted_logits_sum_exp_logits[split_at:]
    exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
    return calculate_loss(predicted_logits, sum_exp_logits, logits_max, compute_z_loss)


fused_calculate_gradients = torch.jit.script(calculate_gradients)


class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, tp_group, compute_z_loss):
        ctx.logits_dtype = vocab_parallel_logits.dtype
        vocab_parallel_logits, logits_max = calculate_logits_max(vocab_parallel_logits)
        torch.distributed.all_reduce(
            logits_max, op=torch.distributed.ReduceOp.MAX, group=tp_group
        )

        partition_vocab_size = vocab_parallel_logits.size(-1)
        rank = torch.distributed.get_rank(group=tp_group)
        world_size = torch.distributed.get_world_size(group=tp_group)
        (
            vocab_start_index,
            vocab_end_index,
        ) = VocabUtility.vocab_range_from_per_partition_vocab_size(
            partition_vocab_size, rank, world_size
        )

        (
            target_mask,
            masked_target_1d,
            predicted_logits_sum_exp_logits,
            exp_logits,
        ) = calculate_predicted_logits(
            vocab_parallel_logits,
            target,
            logits_max,
            vocab_start_index,
            vocab_end_index,
        )

        torch.distributed.all_reduce(
            predicted_logits_sum_exp_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=tp_group,
        )

        loss, penalty, z_loss_grad = calculate_cross_entropy_loss(
            exp_logits, predicted_logits_sum_exp_logits, logits_max, compute_z_loss
        )
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d, z_loss_grad)
        return loss, penalty

    @staticmethod
    def backward(ctx, grad_output, grad_penalty):
        softmax, target_mask, masked_target_1d, z_loss_grad = ctx.saved_tensors
        grad_input = fused_calculate_gradients(
            softmax,
            grad_output,
            target_mask,
            masked_target_1d,
            ctx.logits_dtype,
            z_loss_grad,
            grad_penalty,
        )
        return grad_input, None, None, None


def fused_vocab_parallel_cross_entropy(
    vocab_parallel_logits, target, tp_group, z_loss=0.0, return_components=False
):
    """Return the per-token objective or the separate CE and raw Z-loss over ``tp_group``."""
    loss, penalty = _VocabParallelCrossEntropy.apply(
        vocab_parallel_logits, target, tp_group, z_loss != 0.0
    )
    if return_components:
        return loss, penalty
    return loss if penalty is None else loss + z_loss * penalty
