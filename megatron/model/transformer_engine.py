import torch

try:
    import transformer_engine as te
except ImportError:
    raise ImportError(
        "Unable to import transformer-engine. Please refer to "
        "https://github.com/NVIDIA/TransformerEngine for installation instructions."
    )


class TERMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-8, **kwargs):
        """
            A conditional wrapper to initialize an instance of Transformer-Engine's
            `RMSNorm` based on input
        :param dim: model size
        :param eps:  epsilon value, default 1e-8
        """
        super(TERMSNorm, self).__init__()

        self.d = dim
        self.eps = eps
        self.norm = te.pytorch.RMSNorm(
            hidden_size=self.d,
            eps=self.eps,
            **kwargs,
        )

    def forward(self, x):
        return self.norm(x)


class TELayerNorm(torch.nn.Module):
    def __init__(self, dim, eps=1.0e-5, **kwargs):
        """
            A conditional wrapper to initialize an instance of Transformer-Engine's
            `LayerNorm` based on input
        :param dim: model size
        :param eps:  epsilon value, default 1.0e-5
        """
        super(TELayerNorm, self).__init__()

        self.d = dim
        self.eps = eps
        self.norm = te.pytorch.LayerNorm(
            hidden_size=self.d,
            eps=self.eps,
            **kwargs,
        )

    def forward(self, x):
        return self.norm(x)


class TELinear(te.pytorch.Linear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer.
    """

    def __init__(self):
        # TODO
        return

    def forward(self, x):
        # TODO
        return


class TELayerNormColumnParallelLinear(te.pytorch.LayerNormLinear):
    """
    Wrapper for the Transformer-Engine's `LayerNormLinear` layer that combines
    layernorm and linear layers
    """

    def __init__(self):
        # TODO
        return

    def forward(self, x):
        # TODO
        return


class TEColumnParallelLinear(TELinear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `ColumnParallelLinear` layer.
    """

    def __init__(self):
        # TODO
        return

    def forward(self, x):
        # TODO
        return


class TERowParallelLinear(TELinear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `RowParallelLinear` layer.
    """

    def __init__(self):
        # TODO
        return

    def forward(self, x):
        # TODO
        return


class TEDotProductAttention(te.pytorch.DotProductAttention):
    """
    Wrapper for the Transformer-Engine's `DotProductAttention` layer that also
    has "flash attention" enabled.
    """

    def __init__(self):
        # TODO
        return

    def forward(self, x):
        # TODO
        return


class TEDelayedScaling(te.common.recipe.DelayedScaling):
    """
    Wrapper for the Transformer-Engine's `DelayedScaling` layer.
    """

    def __init__(self):
        # TODO
        return
