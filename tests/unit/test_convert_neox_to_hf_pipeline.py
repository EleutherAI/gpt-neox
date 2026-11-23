import pytest
import torch
from transformers import GPTNeoXForCausalLM

from tools.ckpts import convert_neox_to_hf

HIDDEN = 8
VOCAB = 512  # CharLevelTokenizer vocab size


def make_block():
    shapes = {
        "input_layernorm.weight": (HIDDEN,),
        "input_layernorm.bias": (HIDDEN,),
        "post_attention_layernorm.weight": (HIDDEN,),
        "post_attention_layernorm.bias": (HIDDEN,),
        "attention.query_key_value.weight": (3 * HIDDEN, HIDDEN),
        "attention.query_key_value.bias": (3 * HIDDEN,),
        "attention.dense.weight": (HIDDEN, HIDDEN),
        "attention.dense.bias": (HIDDEN,),
        "mlp.linear1.weight": (4 * HIDDEN, HIDDEN),
        "mlp.linear1.bias": (4 * HIDDEN,),
        "mlp.linear2.weight": (HIDDEN, 4 * HIDDEN),
        "mlp.linear2.bias": (HIDDEN,),
    }
    return {key: torch.randn(shape) for key, shape in shapes.items()}


def save_layer(path, layer_idx, state_dict):
    torch.save(state_dict, path / f"layer_{layer_idx:02}-model_00-model_states.pt")


@pytest.mark.parametrize("num_layers", [1, 2])
def test_convert_pipeline_checkpoint(tmp_path, num_layers):
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()

    # Pipeline layout: embedding at 0, transformer blocks from 2, final norm after them.
    embedding = torch.randn(VOCAB, HIDDEN)
    save_layer(ckpt_dir, 0, {"word_embeddings.weight": embedding})
    blocks = [make_block() for _ in range(num_layers)]
    for i, block in enumerate(blocks):
        save_layer(ckpt_dir, i + 2, block)
    final_norm = {"norm.weight": torch.randn(HIDDEN), "norm.bias": torch.randn(HIDDEN)}
    save_layer(ckpt_dir, num_layers + 3, final_norm)

    config = {
        "num-layers": num_layers,
        "hidden-size": HIDDEN,
        "num-attention-heads": 2,
        "max-position-embeddings": 16,
        "model-parallel-size": 1,
        "pipe-parallel-size": 1,
        "tokenizer-type": "CharLevelTokenizer",
        "norm": "layernorm",
    }

    hf_model = convert_neox_to_hf.convert(
        str(ckpt_dir), config, str(tmp_path / "hf"), sequential=False, precision="fp32"
    )
    hf_model.save_pretrained(tmp_path / "hf")
    loaded, info = GPTNeoXForCausalLM.from_pretrained(
        tmp_path / "hf", output_loading_info=True
    )
    assert not any(info.values()), info

    hf_state = loaded.state_dict()
    assert torch.equal(hf_state["gpt_neox.embed_in.weight"], embedding)
    assert torch.equal(hf_state["embed_out.weight"], embedding)
    assert torch.equal(
        hf_state["gpt_neox.final_layer_norm.weight"], final_norm["norm.weight"]
    )
    keys = convert_neox_to_hf.MODEL_KEYS["neox"]["new"]
    mapping = {
        **keys["COLUMN_PARALLEL_LINEAR_KEYS"],
        **keys["ROW_PARALLEL_LINEAR_KEYS"],
        **keys["ROW_PARALLEL_BIAS_KEYS"],
        **keys["NORM_KEYS"],
    }
    for i, block in enumerate(blocks):
        for neox_key, hf_key in mapping.items():
            assert torch.equal(
                hf_state[f"gpt_neox.layers.{i}.{hf_key}"], block[neox_key]
            ), (i, neox_key)
