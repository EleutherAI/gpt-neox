# Copyright (c) 2021, EleutherAI
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

import torch
import torch.nn.functional as F

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def get_activation(neox_args):
    """retrieves the activation function specified in neox_args"""
    if neox_args.activation == "geglu":
        activation_func = GEGLU(neox_args=neox_args)
    elif neox_args.activation == "gelu":
        if neox_args.onnx_safe and neox_args.bias_gelu_fusion:
            raise ValueError("onnx_safe + bias_gelu_fusion not compatible")
        if neox_args.onnx_safe:
            activation_func = erf_gelu
        elif neox_args.bias_gelu_fusion:
            activation_func = bias_gelu_impl
        else:
            activation_func = F.gelu
    elif neox_args.activation == "relu":
        activation_func = F.relu
    elif neox_args.activation == "softsign":
        activation_func = F.softsign
    elif neox_args.activation == "swish":
        activation_func = swish
    elif neox_args.activation == "mish":
        activation_func = mish
    else:
        raise ValueError(f"Activation function {neox_args.activation} not recognized")
    return activation_func


###### BIAS GELU FUSION/ NO AUTOGRAD ################
# 1/sqrt(2*pi)-> 0.3989423
# 1/sqrt(2)   -> 0.70710678
# sqrt(2/pi)  -> 0.79788456
# this function is tanh approximation of gelu
# actual gelu is:
# x * 0.5 * (1.0 + torch.erf(x * 0.70710678))


@torch.jit.script
def bias_gelu(bias, y):
    x = bias + y
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@torch.jit.script
def bias_gelu_back(g, bias, y):
    x = bias + y
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * (
        (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)
    return ff * g


class GeLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return bias_gelu(bias, input)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = bias_gelu_back(grad_output, bias, input)
        return tmp, tmp


bias_gelu_impl = GeLUFunction.apply


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
    def __init__(self, neox_args):
        super(GEGLU, self).__init__()
        if neox_args.onnx_safe:
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
