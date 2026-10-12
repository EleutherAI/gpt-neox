import os

import pytest
import torch
import yaml
from transformers import AutoModelForCausalLM

from tools.ckpts import convert_neox_to_hf
from tests.common import simulate_deepy_env, save_random_model
from megatron.neox_arguments.neox_args import NeoXArgsTokenizer


@pytest.mark.skip(
    reason="Conversion test is skipped until we fix the CUDA + torch multiprocessing issue."
)
def test_gpt_neox_to_huggingface(monkeypatch, tmpdir, tmp_path):
    # Generate random GPT-NEOX model, check we can convert to hf format

    model_dir = str(tmpdir)
    input_args = ["train.py", "tests/config/test_setup.yml"]
    deepspeed_main_args = simulate_deepy_env(monkeypatch, input_args)
    save_random_model(deepspeed_main_args, model_dir, train_iters=1)

    # Generate output
    script_args = [
        "--config_file",
        "tests/config/test_setup.yml",
        "--input_dir",
        model_dir + "/global_step1",
        "--output_dir",
        model_dir,
    ]
    overwrite_values = {"tokenizer_type": NeoXArgsTokenizer.tokenizer_type}
    convert_neox_to_hf.main(input_args=script_args, overwrite_values=overwrite_values)


def make_neox_config(norm, num_layers=2, hidden_size=64, intermediate_size=256):
    """A minimal NeoX yaml config for a tiny Sequential (pipe-parallel-size: 0) model."""
    return {
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_attention_heads": 4,
        "max_position_embeddings": 128,
        "seq_length": 128,
        "norm": norm,
        "pos_emb": "rotary",
        "activation": "gelu",
        "no_weight_tying": True,
        "pipe_parallel_size": 0,
        "model_parallel_size": 1,
        "make_vocab_size_divisible_by": 128,
        "tokenizer_type": "CharLevelTokenizer",  # vocab size 512, needs no vocab file
    }


def save_synthetic_sequential_checkpoint(checkpoint_dir, neox_config):
    """Construct a random Sequential (pipe-parallel-size: 0) checkpoint directly, so that
    conversion can be tested without CUDA or a training run. Norm parameter names follow
    megatron.model.norms: `weight`/`bias` for layernorm, `scale` for rmsnorm."""
    num_layers = neox_config["num_layers"]
    hidden_size = neox_config["hidden_size"]
    intermediate_size = neox_config["intermediate_size"]
    vocab_size = 512  # CharLevelTokenizer

    state_dict = {
        "sequential.0.word_embeddings.weight": torch.randn(vocab_size, hidden_size)
    }
    for layer_idx in range(2, 2 + num_layers):
        prefix = f"sequential.{layer_idx}."
        state_dict[prefix + "attention.query_key_value.weight"] = torch.randn(
            3 * hidden_size, hidden_size
        )
        state_dict[prefix + "attention.query_key_value.bias"] = torch.randn(
            3 * hidden_size
        )
        state_dict[prefix + "attention.dense.weight"] = torch.randn(
            hidden_size, hidden_size
        )
        state_dict[prefix + "attention.dense.bias"] = torch.randn(hidden_size)
        state_dict[prefix + "mlp.linear1.weight"] = torch.randn(
            intermediate_size, hidden_size
        )
        state_dict[prefix + "mlp.linear1.bias"] = torch.randn(intermediate_size)
        state_dict[prefix + "mlp.linear2.weight"] = torch.randn(
            hidden_size, intermediate_size
        )
        state_dict[prefix + "mlp.linear2.bias"] = torch.randn(hidden_size)
        if neox_config["norm"] == "rmsnorm":
            state_dict[prefix + "input_layernorm.scale"] = torch.randn(hidden_size)
            state_dict[prefix + "post_attention_layernorm.scale"] = torch.randn(
                hidden_size
            )
        else:
            state_dict[prefix + "input_layernorm.weight"] = torch.randn(hidden_size)
            state_dict[prefix + "input_layernorm.bias"] = torch.randn(hidden_size)
            state_dict[prefix + "post_attention_layernorm.weight"] = torch.randn(
                hidden_size
            )
            state_dict[prefix + "post_attention_layernorm.bias"] = torch.randn(
                hidden_size
            )
    if neox_config["norm"] == "rmsnorm":
        state_dict[f"sequential.{num_layers + 3}.norm.scale"] = torch.randn(hidden_size)
    else:
        state_dict[f"sequential.{num_layers + 3}.norm.weight"] = torch.randn(
            hidden_size
        )
        state_dict[f"sequential.{num_layers + 3}.norm.bias"] = torch.randn(hidden_size)
    state_dict[f"sequential.{num_layers + 4}.final_linear.weight"] = torch.randn(
        vocab_size, hidden_size
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(
        {"module": state_dict},
        os.path.join(checkpoint_dir, "mp_rank_00_model_states.pt"),
    )
    return state_dict


def run_neox_to_hf_main(tmpdir, neox_config):
    """Helper running convert_neox_to_hf.main() on a synthetic checkpoint. Returns the
    source state dict and the HF output dir."""
    model_dir = str(tmpdir)
    checkpoint_dir = os.path.join(model_dir, "global_step1")
    output_dir = os.path.join(model_dir, "hf_model")
    config_file = os.path.join(model_dir, "config.yml")
    with open(config_file, "w") as f:
        yaml.dump(neox_config, f)
    state_dict = save_synthetic_sequential_checkpoint(checkpoint_dir, neox_config)

    script_args = [
        "--config_file",
        config_file,
        "--input_dir",
        checkpoint_dir,
        "--output_dir",
        output_dir,
        "--no_save_tokenizer",
    ]
    convert_neox_to_hf.main(input_args=script_args)
    return state_dict, output_dir


@pytest.mark.cpu
def test_gpt_neox_to_huggingface_sequential_layernorm(tmpdir):
    # check that a layernorm Sequential checkpoint converts, and weights survive the round trip
    state_dict, output_dir = run_neox_to_hf_main(
        tmpdir, make_neox_config(norm="layernorm")
    )

    hf_model = AutoModelForCausalLM.from_pretrained(output_dir)
    assert torch.equal(
        hf_model.gpt_neox.embed_in.weight,
        state_dict["sequential.0.word_embeddings.weight"],
    )
    assert torch.equal(
        hf_model.gpt_neox.layers[0].mlp.dense_h_to_4h.weight,
        state_dict["sequential.2.mlp.linear1.weight"],
    )
    assert torch.equal(
        hf_model.gpt_neox.layers[0].input_layernorm.weight,
        state_dict["sequential.2.input_layernorm.weight"],
    )
    assert torch.equal(
        hf_model.gpt_neox.final_layer_norm.weight,
        state_dict["sequential.5.norm.weight"],
    )


@pytest.mark.cpu
def test_gpt_neox_to_huggingface_rmsnorm_raises(tmpdir):
    # regression test for https://github.com/EleutherAI/gpt-neox/issues/1323 :
    # rmsnorm checkpoints store norm params as `scale`, which HF's GPTNeoXForCausalLM cannot
    # represent. conversion used to die with an opaque KeyError partway through; it should
    # raise a clear error up front instead.
    with pytest.raises(ValueError, match="rmsnorm"):
        run_neox_to_hf_main(tmpdir, make_neox_config(norm="rmsnorm"))


@pytest.mark.cpu
def test_llama_architecture_rejects_layernorm(tmpdir):
    # the llama/mistral HF classes only support rmsnorm; mismatched norms should be
    # rejected before any checkpoint is loaded.
    with pytest.raises(ValueError, match="rmsnorm"):
        convert_neox_to_hf.convert(
            os.path.join(str(tmpdir), "does_not_exist"),
            make_neox_config(norm="layernorm"),
            os.path.join(str(tmpdir), "hf_model"),
            sequential=True,
            architecture="llama",
        )
