"""Opt-in cast-and-add for DeepSpeed's existing FP32 accumulation buffers.

The source BF16_Optimizer materializes grad.to(float32) before add_. TensorIterator
can perform that exact widening conversion inside the FP32 add kernel, avoiding
the temporary allocation and its extra memory traffic. No accumulation dtype,
loss scaling, reduction or optimizer state is changed.
"""
from types import MethodType

import torch


@torch.no_grad()
def cast_add_gradient(self, lp, group_idx, param_idx, clear_lp_grads):
    if lp.grad is None:
        return
    hp_grad = self.fp32_groups_gradients[group_idx][param_idx]
    if hp_grad is None or hp_grad.dtype != torch.float32:
        raise RuntimeError('Cast-and-add requires an existing FP32 accumulation buffer')
    if lp.grad.dtype not in (torch.bfloat16, torch.float32):
        raise RuntimeError('Cast-and-add supports BF16/FP32 input gradients only')
    hp_grad.add_(lp.grad.detach().view(hp_grad.shape))
    lp._hp_grad = hp_grad
    self.fp32_groups_has_gradients[group_idx][param_idx] = True
    if clear_lp_grads:
        lp.grad.zero_()


def validate_optimizer(optimizer):
    from deepspeed.runtime.bf16_optimizer import BF16_Optimizer
    if not isinstance(optimizer, BF16_Optimizer):
        raise ValueError('BF16 accumulation optimizations require DeepSpeed BF16_Optimizer')
    if optimizer.grad_acc_dtype != torch.float32:
        raise ValueError('BF16 accumulation optimizations require FP32 gradient accumulation')
    if optimizer.graph_harvesting:
        raise ValueError('Cast-and-add has not been validated with graph harvesting')


def install(optimizer):
    validate_optimizer(optimizer)
    optimizer._update_hp_grad = MethodType(cast_add_gradient, optimizer)
