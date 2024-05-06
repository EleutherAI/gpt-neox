"""Adapter to expose MegaBlocks package, if available."""

try:
    import megablocks
except ImportError:
    megablocks = None


def megablocks_is_available():
    return megablocks is not None


def assert_megablocks_is_available():
    assert (
        megablocks_is_available()
    ), "MegaBlocks not available. Please run `pip install megablocks`."


moe = megablocks.layers.moe if megablocks_is_available() else None
dmoe = megablocks.layers.dmoe if megablocks_is_available() else None
arguments = megablocks.layers.arguments if megablocks_is_available() else None


def as_megablocks_args(neox_args):
    import copy

    tmp = copy.copy(neox_args)
    delattr(tmp, "mlp_type")
    tmp.mlp_type = "mlp"
    args = arguments.from_megatron(tmp)
    args.moe_lbl_in_fp32 = True
    args.fp16 = neox_args.precision == "fp16"
    args.moe_loss_weight = neox_args.moe_loss_coeff
    return args
