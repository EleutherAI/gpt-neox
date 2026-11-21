# Copyright (c) 2025, EleutherAI
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

import math

import torch


def _normal_(tensor, std, truncation=None):
    if truncation is None:
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)
    bound = truncation * std
    return torch.nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-bound, b=bound)


def init_method_normal(sigma, truncation=None):
    """Init method based on N(0, sigma), optionally truncated at +/- truncation * sigma."""

    def init_(tensor):
        return _normal_(tensor, sigma, truncation)

    return init_


def scaled_init_method_normal(
    sigma, num_layers, num_residuals_per_layer=2, truncation=None
):
    """Init method based on N(0, sigma/sqrt(2*num_layers).

    Also allows for N(0, sigma/sqrt(x*num_layers)) where
    x=number of residuals per layer (e.g. 1 for Mamba.)
    """
    return init_method_normal(
        sigma / math.sqrt(num_residuals_per_layer * num_layers), truncation
    )


def fan_in_init_method(truncation=None):
    """N(0, 1/fan_in), with fan_in taken from the full weight when it is sharded along its input dimension."""

    def init_(tensor):
        from megatron import mpu

        fan_in = tensor.shape[1]
        if getattr(tensor, "partition_dim", None) == 1:
            fan_in *= mpu.get_model_parallel_world_size()
        return _normal_(tensor, fan_in**-0.5, truncation)

    return init_


def scaled_init_method(init_method, scale):
    """Multiplies the output of init_method by scale."""

    def init_(tensor):
        init_method(tensor)
        with torch.no_grad():
            return tensor.mul_(scale)

    return init_


# orthogonal init does not support fp16, so have to patch it
def _orthogonal(tensor, gain=1):

    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor.numel() // rows
    flattened = tensor.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    dt = flattened.dtype
    flattened = flattened.to(torch.float32)  # orthogonal init does not support fp16
    q, r = torch.linalg.qr(flattened)
    q, r = q.to(dtype=dt), r.to(dtype=dt)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph

    if rows < cols:
        q.t_()

    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor


def orthogonal_init_method(n_layers=1):
    """Fills the input Tensor with a (semi) orthogonal matrix, as described in
    Exact solutions to the nonlinear dynamics of learning in deep linear neural networks - Saxe, A. et al. (2013)
    Optionally scaling by number of layers possible, as introduced in OBST - Nestler et. al. (2021, to be released)"""

    def init_(tensor):
        return _orthogonal(tensor, math.sqrt(2 / n_layers))

    return init_


def xavier_uniform_init_method():
    """Fills the input Tensor with values according to the method described in Understanding the difficulty of
    training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010), using a uniform distribution."""
    return torch.nn.init.xavier_uniform_


def xavier_normal_init_method():
    """Fills the input Tensor with values according to the method described in Understanding the difficulty of
    training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010), using a normal distribution."""
    return torch.nn.init.xavier_normal_


def small_init_init_method(dim, truncation=None):
    """Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2019), using a normal distribution."""
    return init_method_normal(math.sqrt(2 / (5 * dim)), truncation)


def wang_init_method(n_layers, dim, truncation=None):
    return init_method_normal(2 / n_layers / math.sqrt(dim), truncation)


def get_init_methods(args, hidden_size=None, num_layers=None, layer_number=None):
    """Returns (init_method, output_layer_init_method). hidden_size and num_layers default to the
    model's and are overridden to evaluate width- and depth-dependent inits at a mup base size.
    layer_number is required by layer_scaled_normal."""
    hidden_size = hidden_size or args.hidden_size
    num_layers = num_layers or args.num_layers
    truncation = args.init_truncation

    def _get(name):
        if name == "normal":
            return init_method_normal(args.init_method_std, truncation)
        elif name == "scaled_normal":
            return scaled_init_method_normal(
                args.init_method_std, num_layers, truncation=truncation
            )
        elif name == "layer_scaled_normal":
            return init_method_normal(
                args.init_method_std / math.sqrt(2 * (layer_number + 1)), truncation
            )
        elif name == "fan_in":
            return fan_in_init_method(truncation)
        elif name == "zero":
            return torch.nn.init.zeros_
        elif name == "orthogonal":
            return orthogonal_init_method()
        elif name == "scaled_orthogonal":
            return orthogonal_init_method(num_layers)
        elif name == "xavier_uniform":
            return xavier_uniform_init_method()
        elif name == "xavier_normal":
            return xavier_normal_init_method()
        elif name == "wang_init":
            return wang_init_method(num_layers, hidden_size, truncation)
        elif name == "small_init":
            return small_init_init_method(hidden_size, truncation)
        elif name == "single_residual_scaled_normal":
            # mamba init uses scaled_normal but no need for 2 * num_layers
            # since only one residual per layer
            return scaled_init_method_normal(
                args.init_method_std,
                num_layers,
                num_residuals_per_layer=1,
                truncation=truncation,
            )
        else:
            raise NotImplementedError(f"Unknown init method {name}")

    return _get(args.init_method), _get(args.output_layer_init_method)
