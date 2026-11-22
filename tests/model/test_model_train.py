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

"""
Instantiate models, run for a small number of iterations, and check that training loss improves.

Performs testing using a linear grid search over important parameter values, so that each setting that differs from the base is tested in isolation.

Potentially use fuzzing to test parameters in combination.
"""
import pytest
import torch
import train
from contextlib import nullcontext
from copy import deepcopy
from unittest.mock import patch
from megatron import mpu
from megatron.model.gpt2_model import gpt2_attention_mask_func
from megatron.model.transformer import ParallelTransformerLayer
from megatron.neox_arguments import NeoXArgs
from tests.common import simulate_deepy_env, BASE_CONFIG

PARAMS_TO_TEST = {
    "gpt_j_residual": [True, False],
    "pos_emb": ["learned", "rotary", "sinusoidal", "rpe", "alibi", "none"],
    "attention_config": [
        "global",
        "local",
        "sparse_fixed",
        "sparse_variable",
        "bigbird",
        "bslongformer",
        "gmlp",
        "flash",
    ],
    "hidden_dropout": [0, 0.1],
    "weight_decay": [0, 0.1],
    "use_bias_in_attn_linear": ["all", "none", "qkv"],
    "use_bias_in_norms": [True, False],
    "precision": ["fp16", "fp32", "bfloat16"],
}

keys_to_test = PARAMS_TO_TEST.keys()


@pytest.fixture
def make_attention_layer(monkeypatch):
    monkeypatch.setattr(mpu.initialize, "_MPU_WORLD_SIZE", 1)
    monkeypatch.setattr(mpu.initialize, "_MPU_RANK", 0)
    monkeypatch.setattr(mpu.get_cuda_rng_tracker(), "fork", nullcontext)

    def build(**overrides):
        config = deepcopy(BASE_CONFIG)
        config.update(
            global_num_gpus=1,
            use_cpu_initialization=True,
            hidden_size=32,
            seq_length=16,
            pos_emb="none",
            precision="fp32",
        )
        config.update(overrides)
        init = torch.nn.init.xavier_normal_
        return ParallelTransformerLayer(
            NeoXArgs.from_dict(config),
            gpt2_attention_mask_func,
            init,
            init,
            layer_number=0,
        )

    return build


@pytest.mark.cpu
@pytest.mark.parametrize(
    "mode, kv_heads, qkv_bias, output_bias",
    [
        ("all", 4, True, True),
        ("none", 4, False, False),
        ("qkv", 4, True, False),
        ("qkv", 2, True, False),
    ],
)
def test_gpt_j_attention_bias(
    make_attention_layer, mode, kv_heads, qkv_bias, output_bias
):
    layer = make_attention_layer(
        use_bias_in_attn_linear=mode,
        num_kv_heads=kv_heads,
        gpt_j_residual=True,
        hidden_dropout=1.0,
    )
    assert (layer.attention.query_key_value.bias is not None) == qkv_bias
    assert (layer.attention.dense.bias is not None) == output_bias

    # Isolate the attention contribution to the parallel residual.
    with torch.no_grad():
        layer.mlp.linear2.weight.zero_()
    x = torch.randn(4, 1, layer.neox_args.hidden_size)
    mask = torch.ones(4, 4, dtype=torch.bool).triu(1)
    torch.testing.assert_close(layer(x, mask), x)
    layer.eval()
    attention, bias = layer.attention(layer.input_layernorm(x), mask)
    if bias is not None:
        attention = attention + bias
    torch.testing.assert_close(layer(x, mask), x + attention)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "backend, mode, mlp_config",
    [
        ("te_rowparallel", "qkv", {}),
        ("te_mha", "none", {}),
        ("te_mha", "all", {}),
        (
            "te_columnparallel",
            "none",
            {"use_bias_in_mlp": False, "activation": "swiglu", "intermediate_size": 96},
        ),
    ],
)
def test_te_attention_bias(make_attention_layer, monkeypatch, backend, mode, mlp_config):
    pytest.importorskip("transformer_engine")
    from megatron.model import transformer_engine

    monkeypatch.setattr(
        transformer_engine, "get_tensor_model_parallel_group", lambda: None
    )
    monkeypatch.setattr(transformer_engine, "get_cuda_rng_tracker", None)
    layer = make_attention_layer(
        use_bias_in_attn_linear=mode,
        precision="fp16",
        **{backend: True},
        **mlp_config,
    )
    layer.to(device="cuda", dtype=torch.float16)
    x = torch.randn(16, 1, 32, device="cuda", dtype=torch.float16, requires_grad=True)
    mask = torch.ones(1, 1, 16, 16, dtype=torch.bool, device="cuda").triu(1)
    _, bias = layer.attention(x, mask)
    assert (bias is not None) == (mode == "all")
    layer(x, mask).sum().backward()
    assert torch.isfinite(x.grad).all()


# TODO: fix model training tests
@pytest.mark.skip(
    reason="All model tests are skipped until we fix the CUDA + torch multiprocessing issue."
)
@pytest.mark.parametrize(
    "key, value",
    [(key, value) for key in keys_to_test for value in PARAMS_TO_TEST[key]],
)
def test_model_training_options(monkeypatch, key, value):
    # TODO: Possibly add testing over world_size=2 back in
    neox_args = NeoXArgs.from_dict(BASE_CONFIG)
    if getattr(neox_args, key) == value:
        pytest.skip("Skipping to avoid redundancy as no change in base config")
    if key == "precision" and value == "bfloat16":
        pytest.xfail(
            reason="Assumes that ZeRO optimization stage has been set in the YAML"
        )
    param_dict = {key: value}
    run_train_test(monkeypatch, overwrite_values=param_dict)


def run_train_test(monkeypatch, overwrite_values: dict):
    max_train_iters = 32
    overwrite_values = {**overwrite_values, "train_iters": max_train_iters}
    input_args = ["train.py", "tests/config/test_setup.yml"]
    deepspeed_main_args = simulate_deepy_env(monkeypatch, input_args)

    # Train model, whilst patching collect_loss_for_unit_test to track model loss at each step
    loss_per_iteration = []
    with patch(
        "megatron.training.collect_loss_for_unit_test",
        side_effect=lambda x: loss_per_iteration.append(x),
    ):
        train.main(input_args=deepspeed_main_args, overwrite_values=overwrite_values)
        assert (
            len(loss_per_iteration) == max_train_iters
        ), "patching should have collected loss values from each train step"

        # loss should have decreased by now (otherwise increasing the max_steps parameter could have the testcase pass)
        assert min(loss_per_iteration) < loss_per_iteration[0], (
            "training loss should improve within " + str(max_train_iters) + " steps"
        )
