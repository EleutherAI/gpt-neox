import torch
import torch.nn.functional as F
from typing import Optional
from torch import Tensor


def bias_dropout_add(
    x: Tensor, bias: Tensor, residual: Optional[Tensor], prob: float, training: bool
) -> Tensor:
    out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
    if residual is not None:
        out = residual + out
    return out


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)

    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(
    x: Tensor, bias: Tensor, residual: Optional[Tensor], prob: float
) -> Tensor:
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(
    x: Tensor, bias: Tensor, residual: Optional[Tensor], prob: float
) -> Tensor:
    return bias_dropout_add(x, bias, residual, prob, False)
