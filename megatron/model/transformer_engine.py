import transformer_engine as te
import torch
from pkg_resources import packaging

_te_version = packaging.version.Version(version("transformer-engine"))


class TENorm:
    """
    A conditional wrapper to initialize an instance of Transformer-Engine's
    `LayerNorm` or `RMSNorm` based on input
    """

    def __new__():
        return
        # TODO ???


class TELinear(te.pytorch.Linear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer.
    """

    def __init__(self):
        return
        # TODO: Nick

    def forward(self, x):
        return
        # TODO: Nick


class TELayerNormColumnParallelLinear(te.pytorch.LayerNormLinear):
    """
    Wrapper for the Transformer-Engine's `LayerNormLinear` layer that combines
    layernorm and linear layers
    """

    def __init__(self):
        return
        # TODO: Nick

    def forward(self, x):
        return
        # TODO: Nick


class TEColumnParallelLinear(TELinear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `ColumnParallelLinear` layer.
    """

    def __init__(self):
        # TODO: Nick
        return

    def forward(self, x):
        return
        # TODO: Nick


class TERowParallelLinear(TELinear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `RowParallelLinear` layer.
    """

    def __init__(self):
        # TODO: Nick
        return

    def forward(self, x):
        # TODO: Nick
        return


class TEDotProductAttention(te.pytorch.DotProductAttention):
    """
    Wrapper for the Transformer-Engine's `DotProductAttention` layer that also
    has "flash attention" enabled.
    """

    def __init__(self):
        # TODO: tfidia
        return

    def forward(self, x):
        # TODO: tfidia
        return


class TEDelayedScaling(te.common.recipe.DelayedScaling):
    """
    Wrapper for the Transformer-Engine's `DelayedScaling` layer.
    """

    def __init__(self):
        # TODO: ???
        return
