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

import math

import torch

try:
    import mup
except ImportError:
    pass


def init_method_normal(sigma, use_mup_outer=False, mup_init_scale=1.0):
    """Init method based on N(0, sigma)."""

    def init_(tensor, use_mup=use_mup_outer):
        if use_mup:
            mup.init.normal_(tensor, mean=0.0, std=sigma)
            with torch.no_grad():
                tensor.mul_(mup_init_scale)
            return tensor
        else:
            return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method_normal(
    sigma,
    num_layers,
    use_mup_outer=False,
    mup_init_scale=1.0,
    num_residuals_per_layer=2,
):
    """Init method based on N(0, sigma/sqrt(2*num_layers).

    Also allows for N(0, sigma/sqrt(x*num_layers)) where
    x=number of residuals per layer (e.g. 1 for Mamba.)
    """
    std = sigma / math.sqrt(num_residuals_per_layer * num_layers)

    def init_(tensor, use_mup=use_mup_outer):
        if use_mup:
            mup.init.normal_(tensor, mean=0.0, std=std)
            with torch.no_grad():
                tensor.mul_(mup_init_scale)
            return tensor
        else:
            return torch.nn.init.normal_(tensor, mean=0.0, std=std)

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
    q, r = torch.qr(flattened)
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


def orthogonal_init_method(n_layers=1, use_mup=False, mup_init_scale=1.0):
    """Fills the input Tensor with a (semi) orthogonal matrix, as described in
    Exact solutions to the nonlinear dynamics of learning in deep linear neural networks - Saxe, A. et al. (2013)
    Optionally scaling by number of layers possible, as introduced in OBST - Nestler et. al. (2021, to be released)"""

    if use_mup:
        raise ValueError(
            "Orthogonal init needs to be patched to support mup. Disable mup or use a different init method to avoid this error"
        )

    def init_(tensor):
        return _orthogonal(tensor, math.sqrt(2 / n_layers))

    return init_


def xavier_uniform_init_method(use_mup_outer=False, mup_init_scale=1.0):
    """Fills the input Tensor with values according to the method described in Understanding the difficulty of
    training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010), using a uniform distribution."""

    def init_(tensor, use_mup=use_mup_outer):
        if use_mup:
            mup.init.xavier_uniform_(tensor)
            with torch.no_grad():
                tensor.mul_(mup_init_scale)
            return tensor
        else:
            return torch.nn.init.xavier_uniform_(tensor)

    return init_


def xavier_normal_init_method(use_mup_outer=False, mup_init_scale=1.0):
    """Fills the input Tensor with values according to the method described in Understanding the difficulty of
    training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010), using a normal distribution."""

    def init_(tensor, use_mup=use_mup_outer):
        if use_mup:
            mup.init.xavier_normal_(tensor)
            with torch.no_grad():
                tensor.mul_(mup_init_scale)
            return tensor
        else:
            return torch.nn.init.xavier_normal_(tensor)

    return init_


def small_init_init_method(dim, use_mup_outer=False, mup_init_scale=1.0):
    """Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2010), using a normal distribution."""
    std = math.sqrt(2 / (5 * dim))

    def init_(tensor, use_mup=use_mup_outer):
        if use_mup:
            mup.init.normal_(tensor, mean=0.0, std=std)
            with torch.no_grad():
                tensor.mul_(mup_init_scale)
            return tensor
        else:
            return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def wang_init_method(n_layers, dim, use_mup_outer=False, mup_init_scale=1.0):
    std = 2 / n_layers / math.sqrt(dim)

    def init_(tensor, use_mup=use_mup_outer):
        if use_mup:
            mup.init.normal_(tensor, mean=0.0, std=std)
            with torch.no_grad():
                tensor.mul_(mup_init_scale)
            return tensor
        else:
            return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def get_init_methods(args):

    if args.use_mup:
        try:
            import mup
        except ModuleNotFoundError:
            print("Please install mup https://github.com/microsoft/mup")
            raise Exception

    def _get(name):
        if name == "normal":
            return init_method_normal(
                args.init_method_std, args.use_mup, args.mup_init_scale
            )
        elif name == "scaled_normal":
            return scaled_init_method_normal(
                args.init_method_std, args.num_layers, args.use_mup, args.mup_init_scale
            )
        elif name == "orthogonal":
            return orthogonal_init_method(args.use_mup, args.mup_init_scale)
        elif name == "scaled_orthogonal":
            return orthogonal_init_method(
                args.num_layers, args.use_mup, args.mup_init_scale
            )
        elif name == "xavier_uniform":
            return xavier_uniform_init_method(args.use_mup, args.mup_init_scale)
        elif name == "xavier_normal":
            return xavier_normal_init_method(args.use_mup, args.mup_init_scale)
        elif name == "wang_init":
            return wang_init_method(
                args.num_layers, args.hidden_size, args.use_mup, args.mup_init_scale
            )
        elif name == "small_init":
            return small_init_init_method(
                args.hidden_size, args.use_mup, args.mup_init_scale
            )
        elif name == "single_residual_scaled_normal":
            # mamba init uses scaled_normal but no need for 2 * num_layers
            # since only one residual per layer
            return scaled_init_method_normal(
                args.init_method_std,
                args.num_layers,
                args.use_mup,
                args.mup_init_scale,
                num_residuals_per_layer=1,
            )
        else:
            raise NotImplementedError(f"Unknown init method {name}")

    return _get(args.init_method), _get(args.output_layer_init_method)
