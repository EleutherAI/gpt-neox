import torch
import torch.nn.functional as F
from typing import Optional

# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add(
    x: torch.Tensor,
    bias: Optional[torch.Tensor],
    residual: torch.Tensor,
    prob: float,
    training: bool,
) -> torch.Tensor:
    if bias is not None:
        x = x + bias
    out = torch.nn.functional.dropout(x, p=prob, training=training)
    out = residual + out
    return out


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)

    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(
    x: torch.Tensor, bias: Optional[torch.Tensor], residual: torch.Tensor, prob: float
) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(
    x: torch.Tensor, bias: Optional[torch.Tensor], residual: torch.Tensor, prob: float
) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, False)
