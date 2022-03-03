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

import torch
import torch.nn.functional as F


def get_activation(activation_type, onnx_safe=False, fused_gelu=False):
    """retrieves the activation function specified in neox_args"""
    if activation_type == "geglu":
        activation_func = GEGLU(onnx_safe)
    elif activation_type == "gelu":
        if onnx_safe and fused_gelu:
            raise ValueError("onnx_safe + fused gelu not compatible")
        if onnx_safe:
            activation_func = erf_gelu
        elif fused_gelu:
            activation_func = fused_gelu_impl
        else:
            activation_func = F.gelu
    elif activation_type == "relu":
        activation_func = F.relu
    elif activation_type == "softsign":
        activation_func = F.softsign
    elif activation_type == "swish":
        activation_func = swish
    elif activation_type == "mish":
        activation_func = mish
    else:
        raise ValueError(f"Activation function {activation_type} not recognized")
    return activation_func


@torch.jit.script
def fused_gelu(x):
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


@torch.jit.script
def fused_gelu_back(g, x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * (
        (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)
    return ff * g


class GeLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return fused_gelu(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = fused_gelu_back(grad_output, bias, input)
        return tmp, tmp


fused_gelu_impl = GeLUFunction.apply

# This is actually Python equivalent of torch.nn.functional.gelu(), also with type hints for ONNX exporter
@torch.jit.script
def erf_gelu(x):
    return (
        x
        * 0.5
        * (
            torch.erf(x / 1.41421).to(dtype=x.dtype)
            + torch.ones_like(x).to(dtype=x.dtype)
        )
    )


@torch.jit.script
def swish(x, beta: float = 1.0):
    return x * torch.sigmoid(beta * x)


@torch.jit.script
def mish(x):
    return x * torch.tanh(F.softplus(x))


class GEGLU(torch.nn.Module):
    def __init__(self, onnx_safe):
        super(GEGLU, self).__init__()
        if onnx_safe:
            self.activation_func = erf_gelu
        else:
            self.activation_func = F.gelu

    def forward(self, x, bias=None):
        x, gate = x.chunk(2, dim=-1)
        if bias is not None:
            bias_1, bias_2 = bias.chunk(2, dim=-1)
            x = x + bias_1
            gate = gate + bias_2
        intermediate_parallel = self.activation_func(gate)
        return intermediate_parallel * x
