# Copyright (c) 2026, EleutherAI
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
from types import SimpleNamespace

import pytest
import torch

from tools.ckpts import convert_hf_llama_to_neox
from tools.ckpts import convert_neox_to_hf


QKV_MAPPING = {
    "attention.query_key_value.weight": [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
    ],
    "attention.query_key_value.bias": [
        "self_attn.q_proj.bias",
        "self_attn.k_proj.bias",
        "self_attn.v_proj.bias",
    ],
}


def shaped_tensor(shape, offset=0):
    return torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape) + offset


def make_config(
    *, hidden_size, num_attention_heads, num_key_value_heads, head_dim="missing"
):
    values = {
        "hidden_size": hidden_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "num_hidden_layers": 1,
    }
    if head_dim != "missing":
        values["head_dim"] = head_dim
    return SimpleNamespace(**values)


def make_hf_state_dict(hf_config, *, include_bias=True):
    num_q_heads, num_kv_heads, head_dim = (
        convert_hf_llama_to_neox.get_attention_dimensions(hf_config)
    )
    hidden_size = hf_config.hidden_size
    query_hidden_size = num_q_heads * head_dim
    kv_hidden_size = num_kv_heads * head_dim
    intermediate_size = 16
    vocab_size = 32
    prefix = "model.layers.0"
    state_dict = {
        "model.embed_tokens.weight": shaped_tensor((vocab_size, hidden_size), 1),
        f"{prefix}.self_attn.q_proj.weight": shaped_tensor(
            (query_hidden_size, hidden_size), 1000
        ),
        f"{prefix}.self_attn.k_proj.weight": shaped_tensor(
            (kv_hidden_size, hidden_size), 2000
        ),
        f"{prefix}.self_attn.v_proj.weight": shaped_tensor(
            (kv_hidden_size, hidden_size), 3000
        ),
        f"{prefix}.self_attn.o_proj.weight": shaped_tensor(
            (hidden_size, query_hidden_size), 4000
        ),
        f"{prefix}.mlp.gate_proj.weight": shaped_tensor(
            (intermediate_size, hidden_size), 5000
        ),
        f"{prefix}.mlp.up_proj.weight": shaped_tensor(
            (intermediate_size, hidden_size), 6000
        ),
        f"{prefix}.mlp.down_proj.weight": shaped_tensor(
            (hidden_size, intermediate_size), 7000
        ),
        f"{prefix}.input_layernorm.weight": shaped_tensor((hidden_size,), 8000),
        f"{prefix}.post_attention_layernorm.weight": shaped_tensor(
            (hidden_size,), 9000
        ),
        "model.norm.weight": shaped_tensor((hidden_size,), 10000),
        "lm_head.weight": shaped_tensor((vocab_size, hidden_size), 11000),
    }
    if include_bias:
        state_dict.update(
            {
                f"{prefix}.self_attn.q_proj.bias": shaped_tensor(
                    (query_hidden_size,), 12000
                ),
                f"{prefix}.self_attn.k_proj.bias": shaped_tensor(
                    (kv_hidden_size,), 13000
                ),
                f"{prefix}.self_attn.v_proj.bias": shaped_tensor(
                    (kv_hidden_size,), 14000
                ),
                f"{prefix}.self_attn.o_proj.bias": shaped_tensor((hidden_size,), 15000),
            }
        )
    return state_dict


def round_trip_attention(hf_state_dict, hf_config, tp_size):
    converted = convert_hf_llama_to_neox.convert_model(
        hf_state_dict, hf_config, tp_size
    )
    loaded_tp_ranks = [{"module": shard} for shard in converted]
    reconstructed = convert_neox_to_hf.reshard_and_split_qkv(
        param_mapping=QKV_MAPPING,
        hf_config=hf_config,
        loaded_tp_ranks=loaded_tp_ranks,
        layer_idx=2,
        sequential=True,
    )
    return converted, reconstructed


@pytest.mark.cpu
@pytest.mark.parametrize("tp_size", [1, 2])
def test_expanded_attention_projections_round_trip(tp_size):
    hf_config = make_config(
        hidden_size=8,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
    )
    hf_state_dict = make_hf_state_dict(hf_config)

    converted, reconstructed = round_trip_attention(hf_state_dict, hf_config, tp_size)

    qkv_key = "sequential.2.attention.query_key_value.weight"
    qkv_bias_key = "sequential.2.attention.query_key_value.bias"
    dense_key = "sequential.2.attention.dense.weight"
    dense_bias_key = "sequential.2.attention.dense.bias"
    assert converted[0][qkv_key].shape == (4096 // tp_size, 8)
    assert converted[0][qkv_bias_key].shape == (4096 // tp_size,)
    assert converted[0][dense_key].shape == (8, 2048 // tp_size)
    assert converted[0][dense_bias_key].shape == (8,)
    assert torch.equal(
        torch.cat([rank[dense_key] for rank in converted], dim=1),
        hf_state_dict["model.layers.0.self_attn.o_proj.weight"],
    )
    for projection in ("q", "k", "v"):
        for parameter in ("weight", "bias"):
            hf_key = f"model.layers.0.self_attn.{projection}_proj.{parameter}"
            reconstructed_key = f"self_attn.{projection}_proj.{parameter}"
            assert torch.equal(reconstructed[reconstructed_key], hf_state_dict[hf_key])


@pytest.mark.cpu
def test_legacy_attention_projection_widths_round_trip():
    hf_config = make_config(
        hidden_size=8,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    hf_state_dict = make_hf_state_dict(hf_config)

    converted, reconstructed = round_trip_attention(hf_state_dict, hf_config, 2)

    assert converted[0]["sequential.2.attention.query_key_value.weight"].shape == (8, 8)
    assert reconstructed["self_attn.q_proj.weight"].shape == (8, 8)
    assert reconstructed["self_attn.k_proj.weight"].shape == (4, 8)
    assert torch.equal(
        reconstructed["self_attn.q_proj.weight"],
        hf_state_dict["model.layers.0.self_attn.q_proj.weight"],
    )


@pytest.mark.cpu
def test_create_config_propagates_explicit_head_dim(monkeypatch):
    def fake_build_tokenizer(args):
        args.padded_vocab_size = 32
        return SimpleNamespace(pad=0, eod=1)

    monkeypatch.setattr(convert_neox_to_hf, "build_tokenizer", fake_build_tokenizer)
    neox_config = {
        "model-parallel-size": 1,
        "vocab-file": "unused",
        "merge-file": "unused",
        "tokenizer-type": "GPT2BPETokenizer",
        "hidden-size": 8,
        "num-layers": 1,
        "num-attention-heads": 16,
        "num-kv-heads": 8,
        "head-dim": 128,
        "max-position-embeddings": 32,
        "intermediate-size": 16,
        "use-bias-in-attn-linear": False,
    }

    hf_config = convert_neox_to_hf.create_config(neox_config, architecture="llama")

    assert hf_config.head_dim == 128


@pytest.mark.cpu
def test_export_rejects_hf_target_that_ignores_head_dim():
    hf_config = make_config(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=6,
    )
    converted_state_dict = {
        "self_attn.q_proj.weight": torch.empty(12, 8),
        "self_attn.k_proj.weight": torch.empty(6, 8),
        "self_attn.v_proj.weight": torch.empty(6, 8),
        "self_attn.o_proj.weight": torch.empty(8, 12),
    }
    quotient_target = SimpleNamespace(
        state_dict=lambda: {
            "self_attn.q_proj.weight": torch.empty(8, 8),
            "self_attn.k_proj.weight": torch.empty(4, 8),
            "self_attn.v_proj.weight": torch.empty(4, 8),
            "self_attn.o_proj.weight": torch.empty(8, 8),
        }
    )

    with pytest.raises(
        ValueError,
        match="does not honor the configured head_dim",
    ):
        convert_neox_to_hf.validate_hf_attention_projection_shapes(
            quotient_target,
            hf_config,
            converted_state_dict,
            layer_idx=0,
        )


@pytest.mark.cpu
def test_llama_export_wires_optional_attention_biases():
    for naming in ("new", "legacy"):
        mapping = convert_neox_to_hf.MODEL_KEYS["llama"][naming]
        assert mapping["OPTIONAL_GQA_QKV_KEYS"]["attention.query_key_value.bias"] == [
            "self_attn.q_proj.bias",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.bias",
        ]
        assert mapping["OPTIONAL_REPLICATED_KEYS"]["attention.dense.bias"] == (
            "self_attn.o_proj.bias"
        )


@pytest.mark.cpu
def test_export_projection_validation_includes_attention_biases():
    hf_config = make_config(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=6,
    )
    weights = {
        "self_attn.q_proj.weight": torch.empty(12, 8),
        "self_attn.k_proj.weight": torch.empty(6, 8),
        "self_attn.v_proj.weight": torch.empty(6, 8),
        "self_attn.o_proj.weight": torch.empty(8, 12),
    }
    target_state_dict = {
        **weights,
        "self_attn.q_proj.bias": torch.empty(12),
        "self_attn.k_proj.bias": torch.empty(6),
        "self_attn.v_proj.bias": torch.empty(6),
        "self_attn.o_proj.bias": torch.empty(8),
    }
    target = SimpleNamespace(state_dict=lambda: target_state_dict)

    with pytest.raises(ValueError, match="other side does not"):
        convert_neox_to_hf.validate_hf_attention_projection_shapes(
            target,
            hf_config,
            converted_state_dict=weights,
            layer_idx=0,
        )


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("num_attention_heads", "num_key_value_heads", "match"),
    [
        (3, 1, "num_attention_heads.*must be divisible"),
        (4, 1, "num_key_value_heads.*must be divisible"),
    ],
)
def test_import_rejects_heads_that_cannot_be_tp_partitioned(
    num_attention_heads, num_key_value_heads, match
):
    hf_config = make_config(
        hidden_size=8,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=2,
    )

    with pytest.raises(ValueError, match=match):
        convert_hf_llama_to_neox.convert_model({}, hf_config, tp_ranks=2)


@pytest.mark.cpu
def test_import_rejects_incorrect_projection_shape():
    hf_config = make_config(
        hidden_size=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    )
    hf_state_dict = make_hf_state_dict(hf_config)
    hf_state_dict["model.layers.0.self_attn.q_proj.weight"] = torch.empty(8, 8)

    with pytest.raises(
        ValueError,
        match=r"q_proj.weight has shape \(8, 8\), expected \(16, 8\)",
    ):
        convert_hf_llama_to_neox.convert_model(hf_state_dict, hf_config, tp_ranks=1)


@pytest.mark.cpu
def test_export_rejects_incorrect_fused_qkv_shape():
    hf_config = make_config(
        hidden_size=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    )
    loaded_tp_ranks = [
        {
            "module": {
                "sequential.2.attention.query_key_value.weight": torch.empty(31, 8)
            }
        }
    ]

    with pytest.raises(
        ValueError,
        match=r"has shape \(31, 8\), expected \(32, 8\) for head_dim=4",
    ):
        convert_neox_to_hf.reshard_and_split_qkv(
            param_mapping={
                "attention.query_key_value.weight": QKV_MAPPING[
                    "attention.query_key_value.weight"
                ]
            },
            hf_config=hf_config,
            loaded_tp_ranks=loaded_tp_ranks,
            layer_idx=2,
            sequential=True,
        )
