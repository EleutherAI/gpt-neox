import math

import torch


def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method_normal(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
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

    def init_(tensor):
        return torch.nn.init.xavier_uniform_(tensor)

    return init_


def xavier_normal_init_method():
    """Fills the input Tensor with values according to the method described in Understanding the difficulty of
    training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010), using a normal distribution."""

    def init_(tensor):
        return torch.nn.init.xavier_normal_(tensor)

    return init_


def small_init_init_method(dim):
    """Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2010), using a normal distribution."""
    std = math.sqrt(2 / (5 * dim))

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def wang_init_method(n_layers, dim):
    std = 2 / n_layers / math.sqrt(dim)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def get_init_methods(args):
    def _get(name):
        if name == "normal":
            return init_method_normal(args.init_method_std)
        elif name == "scaled_normal":
            return scaled_init_method_normal(args.init_method_std, args.num_layers)
        elif name == "orthogonal":
            return orthogonal_init_method()
        elif name == "scaled_orthogonal":
            return orthogonal_init_method(args.num_layers)
        elif name == "xavier_uniform":
            return xavier_uniform_init_method()
        elif name == "xavier_normal":
            return xavier_normal_init_method()
        elif name == "wang_init":
            return wang_init_method(args.num_layers, args.hidden_size)
        elif name == "small_init":
            return small_init_init_method(args.hidden_size)
        else:
            raise NotImplementedError(f"Unknown init method {name}")

    return _get(args.init_method), _get(args.output_layer_init_method)
