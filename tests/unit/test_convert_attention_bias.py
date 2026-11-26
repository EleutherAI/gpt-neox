import pytest
import torch
from types import SimpleNamespace
from tools.ckpts.convert_neox_to_hf import (
    has_state,
    get_state,
    reshard_and_split_qkv,
    MODEL_KEYS,
)


def test_has_state():
    seq_ranks = [
        {"module": {"sequential.2.weight": torch.ones(1), "sequential.2.bias": torch.zeros(1)}},
        {"module": {"sequential.2.weight": torch.ones(1), "sequential.2.bias": torch.zeros(1)}},
    ]
    assert has_state(seq_ranks, "weight", layer_idx=2, sequential=True) is True
    assert has_state(seq_ranks, "bias", layer_idx=2, sequential=True) is True
    assert has_state(seq_ranks, "missing_key", layer_idx=2, sequential=True) is False

    nonseq_ranks = [
        {"weight": torch.ones(1)},
        {"weight": torch.ones(1)},
    ]
    assert has_state(nonseq_ranks, "weight", layer_idx=2, sequential=False) is True
    assert has_state(nonseq_ranks, "bias", layer_idx=2, sequential=False) is False


def test_llama_attention_bias_reshard_and_split():
    hidden_size = 64
    num_heads = 4
    num_kv_heads = 2
    head_dim = hidden_size // num_heads  # 16
    kv_hidden_size = num_kv_heads * head_dim  # 32

    hf_config = SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
    )

    # 2 TP ranks
    tp_size = 2
    q_per_rank = hidden_size // tp_size  # 32
    k_per_rank = kv_hidden_size // tp_size  # 16
    v_per_rank = kv_hidden_size // tp_size  # 16
    total_qkv_per_rank = q_per_rank + k_per_rank + v_per_rank  # 64

    # Rank 0 and Rank 1 biases
    b0 = torch.arange(total_qkv_per_rank, dtype=torch.float32)
    b1 = torch.arange(total_qkv_per_rank, dtype=torch.float32) + 100

    loaded_tp_ranks = [
        {"module": {"sequential.2.attention.query_key_value.bias": b0}},
        {"module": {"sequential.2.attention.query_key_value.bias": b1}},
    ]

    param_mapping = {
        "attention.query_key_value.bias": [
            "self_attn.q_proj.bias",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.bias",
        ]
    }

    state_dict = reshard_and_split_qkv(
        param_mapping=param_mapping,
        hf_config=hf_config,
        loaded_tp_ranks=loaded_tp_ranks,
        layer_idx=2,
        sequential=True,
    )

    assert "self_attn.q_proj.bias" in state_dict
    assert "self_attn.k_proj.bias" in state_dict
    assert "self_attn.v_proj.bias" in state_dict

    assert state_dict["self_attn.q_proj.bias"].shape == (hidden_size,)
    assert state_dict["self_attn.k_proj.bias"].shape == (kv_hidden_size,)
    assert state_dict["self_attn.v_proj.bias"].shape == (kv_hidden_size,)


def test_llama_model_keys_include_attention_biases():
    for mode in ["new", "legacy"]:
        llama_keys = MODEL_KEYS["llama"][mode]
        assert "attention.dense.bias" in llama_keys["ROW_PARALLEL_BIAS_KEYS"]
        assert llama_keys["ROW_PARALLEL_BIAS_KEYS"]["attention.dense.bias"] == "self_attn.o_proj.bias"

        assert "attention.query_key_value.bias" in llama_keys["GQA_QKV_KEYS"]
        assert llama_keys["GQA_QKV_KEYS"]["attention.query_key_value.bias"] == [
            "self_attn.q_proj.bias",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.bias",
        ]


def test_reshard_and_split_skips_missing_bias():
    hf_config = SimpleNamespace(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    # Checkpoint has only weights, no bias
    loaded_tp_ranks = [
        {"module": {}},
        {"module": {}},
    ]
    param_mapping = {
        "attention.query_key_value.bias": [
            "self_attn.q_proj.bias",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.bias",
        ]
    }
    state_dict = reshard_and_split_qkv(
        param_mapping=param_mapping,
        hf_config=hf_config,
        loaded_tp_ranks=loaded_tp_ranks,
        layer_idx=2,
        sequential=True,
    )
    # When bias is missing in checkpoint, reshard_and_split_qkv gracefully returns empty dict
    assert state_dict == {}
