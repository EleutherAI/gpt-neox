# Copyright (c) 2026, EleutherAI
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, sentinel

import pytest
import torch


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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
    reason="FP8 checkpoint test requires Hopper or newer",
)
@pytest.mark.parametrize("history", [1, 4])
def test_same_version_fp8_checkpoint(integration, args, monkeypatch, history):
    monkeypatch.setattr(integration, "get_tensor_model_parallel_group", lambda: None)
    args.te_fp8_amax_history_len = history
    recipe = integration.TEDelayedScaling(args)
    linear = integration.te.pytorch.Linear
    layer = linear(16, 16, params_dtype=torch.bfloat16, device="cuda")
    inputs = torch.randn_like(layer.weight, requires_grad=True)
    with recipe.get_context():
        output = layer(inputs)
    output.sum().backward()

    restored = linear(16, 16, params_dtype=torch.bfloat16, device="cuda")
    # The state in this test was produced by the layer above.
    monkeypatch.setenv("NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE", "1")
    restored.load_state_dict(layer.state_dict())
    for direction in ("scaling_fwd", "scaling_bwd"):
        original = layer.fp8_meta[direction]
        loaded = restored.fp8_meta[direction]
        assert loaded.amax_history.shape[0] == history
        assert torch.equal(loaded.amax_history, original.amax_history)
        assert torch.equal(loaded.scale, original.scale)


@pytest.fixture
def make_block(integration, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("FP8 attention fixture requires Hopper or newer")
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
    from megatron import mpu
    from megatron.model.gpt2_model import gpt2_attention_mask_func
    from megatron.model.transformer import ParallelTransformerLayer
    from megatron.neox_arguments import NeoXArgs

    monkeypatch.setattr(mpu.initialize, "_MPU_WORLD_SIZE", 1)
    monkeypatch.setattr(mpu.initialize, "_MPU_RANK", 0)
    monkeypatch.setattr(integration, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(mpu.get_cuda_rng_tracker(), "fork", nullcontext)

    def build(fp8, fused_mlp=True, te_linears=False):
        if fp8 and torch.backends.cudnn.version() < 90201:
            pytest.skip("causal FP8 attention fixture needs cuDNN 9.2.1 or newer")
        config = NeoXArgs(
            global_num_gpus=1,
            train_micro_batch_size_per_gpu=2,
            num_layers=1,
            hidden_size=256,
            num_attention_heads=2,
            seq_length=128,
            max_position_embeddings=128,
            precision="bfloat16",
            bf16={"enabled": True},
            zero_optimization={"stage": 1},
            te_mha=True,
            te_fp8_mha=fp8,
            te_fp8_amax_history_len=4,
            te_layernorm_mlp=fused_mlp,
            te_columnparallel=te_linears,
            te_rowparallel=te_linears,
            hidden_dropout=0.2,
            attention_dropout=0.0,
        )
        init = partial(torch.nn.init.normal_, std=0.02)
        return (
            ParallelTransformerLayer(config, gpt2_attention_mask_func, init, init, 0)
            .cuda()
            .bfloat16()
        )

    FP8GlobalStateManager.reset()
    yield build
    FP8GlobalStateManager.reset()


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize(
    "fused_mlp,te_linears", [(True, False), (True, True), (False, True), (False, False)]
)
def test_block_precision_and_eval(make_block, fp8, fused_mlp, te_linears):
    from megatron import mpu
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    block = make_block(fp8, fused_mlp, te_linears)
    x = torch.randn(
        128, 2, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    mask = torch.ones(1, 1, 128, 128, device="cuda", dtype=torch.bool).triu(1)
    output = block(x, mask)
    assert output.dtype == torch.bfloat16
    output.float().square().sum().backward()
    assert x.grad is not None and x.grad.isfinite().all() and x.grad.count_nonzero()

    modules = [block.attention.qkv, block.attention.proj]
    if fused_mlp:
        modules.append(block.mlp)
    elif te_linears:
        modules.extend([block.mlp.linear1, block.mlp.linear2])
    else:
        assert isinstance(block.mlp.linear1, mpu.ColumnParallelLinear)
        assert isinstance(block.mlp.linear2, mpu.RowParallelLinear)
    for module in modules:
        assert module.fp8 == fp8
        assert module.activation_dtype == torch.bfloat16
        assert any(
            p.grad is not None and p.grad.count_nonzero() for p in module.parameters()
        )
        if fp8:
            assert module.fp8_meta["scaling_fwd"].amax_history.count_nonzero()
    if fp8:
        # TE can pass a quantized gradient between adjacent operators.
        assert any(
            m.fp8_meta["scaling_bwd"].amax_history.count_nonzero() for m in modules
        )
    assert not FP8GlobalStateManager.is_fp8_enabled()

    block.eval()
    with torch.no_grad():
        prediction = block(x, mask)
        assert prediction.dtype == torch.bfloat16 and not prediction.requires_grad
    eval_input = x.detach().requires_grad_()
    block(eval_input, mask).float().square().sum().backward()
    assert eval_input.grad is not None and eval_input.grad.isfinite().all()
    assert not FP8GlobalStateManager.is_fp8_enabled()


@pytest.mark.parametrize("use_reentrant", [True, False])
def test_fp8_checkpoint_replays_forward_scales(make_block, integration, use_reentrant):
    from torch.utils.checkpoint import checkpoint
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    # Exercise both autograd modes without setting up distributed training.
    checkpoint_func = partial(checkpoint, use_reentrant=use_reentrant)

    def run(recompute):
        FP8GlobalStateManager.reset()
        torch.manual_seed(1234)
        blocks = [make_block(True), make_block(True)]
        mask = torch.ones(1, 1, 128, 128, device="cuda", dtype=torch.bool).triu(1)
        result = {}
        pending = []
        for microbatch in range(2):
            x = torch.randn(
                128, 2, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            y = x
            for block in blocks:
                if recompute:
                    y = integration.fp8_activation_checkpoint(
                        checkpoint_func, block, y, mask
                    )
                else:
                    y = block(y, mask)
            result[f"output.{microbatch}"] = y.detach()
            pending.append((microbatch, x, y))
        # Each outstanding microbatch needs the scales used by its own forward.
        for microbatch, x, y in reversed(pending):
            y.float().square().mean().backward()
            result[f"input_grad.{microbatch}"] = x.grad
        for i, block in enumerate(blocks):
            for name, parameter in block.named_parameters():
                result[f"{i}.{name}.grad"] = parameter.grad
            for name, module in block.named_modules():
                if getattr(module, "fp8", False) and hasattr(module, "fp8_meta"):
                    for direction in ("scaling_fwd", "scaling_bwd"):
                        state = module.fp8_meta[direction]
                        result[f"{i}.{name}.{direction}.history"] = (
                            state.amax_history.clone()
                        )
                        result[f"{i}.{name}.{direction}.scale"] = state.scale.clone()
        result["rng"] = torch.cuda.get_rng_state()
        return result

    torch.testing.assert_close(run(True), run(False), rtol=0, atol=0)
