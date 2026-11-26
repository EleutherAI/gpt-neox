import pytest
from tools.ckpts.convert_neox_to_hf import get_mlp_naming_convention, MODEL_KEYS


def test_get_mlp_naming_convention_legacy_llama_sequential():
    # Sequential checkpoint where module dictionary holds keys
    loaded_tp_ranks = [
        {
            "module": {
                "sequential.2.attention.query_key_value.weight": None,
                "sequential.2.mlp.w1.weight": None,
                "sequential.2.mlp.w2.weight": None,
                "sequential.2.mlp.w3.weight": None,
            }
        }
    ]
    convention = get_mlp_naming_convention(loaded_tp_ranks, layer_idx=0, sequential=True)
    assert convention == "legacy"


def test_get_mlp_naming_convention_legacy_llama_non_sequential():
    # Non-sequential checkpoint
    loaded_tp_ranks = [
        {
            "attention.query_key_value.weight": None,
            "mlp.w1.weight": None,
            "mlp.w2.weight": None,
            "mlp.w3.weight": None,
        }
    ]
    convention = get_mlp_naming_convention(loaded_tp_ranks, layer_idx=0, sequential=False)
    assert convention == "legacy"


def test_get_mlp_naming_convention_standard_conventions():
    # Transformer engine
    te_ranks = [{"mlp.fc1_weight": None}]
    assert get_mlp_naming_convention(te_ranks, layer_idx=0, sequential=False) == "transformer_engine"

    # New
    new_ranks = [{"mlp.linear1.weight": None}]
    assert get_mlp_naming_convention(new_ranks, layer_idx=0, sequential=False) == "new"

    # Legacy dense_h_to_4h
    legacy_ranks = [{"mlp.dense_h_to_4h.weight": None}]
    assert get_mlp_naming_convention(legacy_ranks, layer_idx=0, sequential=False) == "legacy"


def test_get_mlp_naming_convention_unknown_raises():
    unknown_ranks = [{"mlp.unknown_layer.weight": None}]
    with pytest.raises(ValueError, match="Unable to determine MLP naming convention"):
        get_mlp_naming_convention(unknown_ranks, layer_idx=0, sequential=False)


def test_llama_legacy_key_translation_map():
    # Verify that Llama legacy maps w1 to gate_proj, w2 to down_proj, and w3 to up_proj
    llama_legacy = MODEL_KEYS["llama"]["legacy"]
    col_keys = llama_legacy["COLUMN_PARALLEL_LINEAR_KEYS"]
    row_keys = llama_legacy["ROW_PARALLEL_LINEAR_KEYS"]

    assert col_keys["mlp.w1.weight"] == "mlp.gate_proj.weight"
    assert col_keys["mlp.w3.weight"] == "mlp.up_proj.weight"
    assert row_keys["mlp.w2.weight"] == "mlp.down_proj.weight"
