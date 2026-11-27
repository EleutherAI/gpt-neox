# Copyright (c) 2026, EleutherAI
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock, sentinel

import pytest
import torch
from packaging.version import Version


@pytest.fixture
def integration():
    pytest.importorskip("transformer_engine.pytorch")
    from megatron.model import transformer_engine

    return transformer_engine


@pytest.fixture
def args():
    return SimpleNamespace(
        te_fp8_mha=True,
        te_fp8_format="hybrid",
        te_fp8_margin=1,
        te_fp8_amax_history_len=4,
        te_fp8_amax_compute_algo="most_recent",
        te_fp8_wgrad=True,
    )


def test_delayed_scaling_context(integration, args, monkeypatch):
    wrapper = integration.TEDelayedScaling(args)
    recipe = wrapper.recipe
    assert type(recipe) is integration.te.common.recipe.DelayedScaling
    assert recipe.fp8_mha and recipe.fp8_dpa
    assert recipe.fp8_format == integration.te.common.recipe.Format.HYBRID
    assert (recipe.margin, recipe.amax_history_len, recipe.amax_compute_algo) == (
        1,
        4,
        "most_recent",
    )

    monkeypatch.setattr(
        integration, "get_tensor_model_parallel_group", lambda: sentinel.group
    )
    autocast = Mock()
    if hasattr(integration.te.pytorch, "autocast"):
        monkeypatch.setattr(integration.te.pytorch, "autocast", autocast)
        expected = dict(recipe=recipe, amax_reduction_group=sentinel.group)
    else:
        monkeypatch.setattr(integration.te.pytorch, "fp8_autocast", autocast)
        expected = dict(fp8_recipe=recipe, fp8_group=sentinel.group)
    assert wrapper.get_context() is autocast.return_value
    autocast.assert_called_once_with(enabled=True, **expected)


def test_high_precision_weight_gradients(integration, args):
    args.te_fp8_wgrad = False
    if hasattr(
        integration.te.common.recipe.DelayedScaling, "override_linear_precision"
    ):
        assert integration.TEDelayedScaling(args).recipe.override_linear_precision.wgrad
    else:
        with pytest.raises(ValueError, match="te_fp8_wgrad=False"):
            integration.TEDelayedScaling(args)


def test_disabled_fp8_accepts_high_precision_weight_gradients():
    from megatron.neox_arguments import NeoXArgs

    config = NeoXArgs(
        global_num_gpus=1,
        train_micro_batch_size_per_gpu=1,
        num_layers=1,
        hidden_size=128,
        num_attention_heads=1,
        seq_length=16,
        max_position_embeddings=16,
        te_mha=True,
        te_fp8_mha=False,
        te_fp8_wgrad=False,
    )
    assert not config.te_fp8_wgrad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE allocates scales on CUDA")
@pytest.mark.parametrize("length", [1, 4])
def test_load_legacy_delayed_scaling_history(integration, args, monkeypatch, length):
    layer = torch.nn.Sequential(integration.te.pytorch.Linear(16, 16, device="cuda"))
    wrapper = integration.TEDelayedScaling(args)
    wrapper.register_checkpoint_hook(layer)
    weights = layer.state_dict()
    # TE 1.12 saves these tensors and the GEMM count in a byte stream.
    state = {"extra_fp8_variables": {"num_gemms": 1}}
    for direction, width in (("fwd", 3), ("bwd", 2)):
        state[f"scale_{direction}"] = torch.arange(1, width + 1).float()
        state[f"scale_inv_{direction}"] = state[f"scale_{direction}"].reciprocal()
        state[f"amax_history_{direction}"] = (
            torch.arange(length * width).float().view(length, width)
        )
    buffer = BytesIO()
    torch.save(state, buffer)
    weights["0._extra_state"] = buffer
    if Version(integration.te.__version__) >= Version("2.21"):
        monkeypatch.delenv("NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE", raising=False)
        with pytest.raises(RuntimeError, match="NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE"):
            layer.load_state_dict(weights)
    monkeypatch.setenv("NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE", "1")
    layer.load_state_dict(weights)
    for direction in ("fwd", "bwd"):
        scaling = layer[0].fp8_meta[f"scaling_{direction}"]
        torch.testing.assert_close(scaling.scale.cpu(), state[f"scale_{direction}"])
        torch.testing.assert_close(
            scaling.amax_history.cpu(), state[f"amax_history_{direction}"]
        )
