import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import tqdm


def get_attention_dimensions(hf_config):
    """Resolve attention widths without importing the NeoX model stack."""
    hidden_size = hf_config.hidden_size
    num_q_heads = hf_config.num_attention_heads
    num_kv_heads = getattr(hf_config, "num_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = num_q_heads
    head_dim = getattr(hf_config, "head_dim", None)

    if not isinstance(num_q_heads, int) or num_q_heads <= 0:
        raise ValueError(
            f"num_attention_heads must be a positive integer, got {num_q_heads}"
        )
    if not isinstance(num_kv_heads, int) or num_kv_heads <= 0:
        raise ValueError(
            f"num_key_value_heads must be a positive integer, got {num_kv_heads}"
        )
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            "num_key_value_heads must evenly divide num_attention_heads, got "
            f"{num_kv_heads} and {num_q_heads}"
        )
    if head_dim is None:
        if hidden_size % num_q_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads when "
                f"head_dim is not set, got {hidden_size} and {num_q_heads}"
            )
        head_dim = hidden_size // num_q_heads
    if not isinstance(head_dim, int) or head_dim <= 0:
        raise ValueError(f"head_dim must be a positive integer, got {head_dim}")

    return num_q_heads, num_kv_heads, head_dim


def validate_attention_tp(num_q_heads, num_kv_heads, tp_ranks):
    if not isinstance(tp_ranks, int) or tp_ranks <= 0:
        raise ValueError(f"tp_ranks must be a positive integer, got {tp_ranks}")
    if num_q_heads % tp_ranks != 0:
        raise ValueError(
            f"num_attention_heads ({num_q_heads}) must be divisible by tp_ranks ({tp_ranks})"
        )
    if num_kv_heads % tp_ranks != 0:
        raise ValueError(
            f"num_key_value_heads ({num_kv_heads}) must be divisible by tp_ranks ({tp_ranks})"
        )


def require_shape(state_dict, key, expected_shape):
    if key not in state_dict:
        raise ValueError(f"Missing required attention projection {key}")
    actual_shape = tuple(state_dict[key].shape)
    if actual_shape != tuple(expected_shape):
        raise ValueError(
            f"Attention projection {key} has shape {actual_shape}, expected {tuple(expected_shape)}"
        )


def get_attention_projections(hf_state_dict, hf_config, layer_num, tp_ranks):
    num_q_heads, num_kv_heads, head_dim = get_attention_dimensions(hf_config)
    validate_attention_tp(num_q_heads, num_kv_heads, tp_ranks)

    hidden_size = hf_config.hidden_size
    query_hidden_size = num_q_heads * head_dim
    kv_hidden_size = num_kv_heads * head_dim
    prefix = f"model.layers.{layer_num}.self_attn"
    keys = {
        "q": f"{prefix}.q_proj.weight",
        "k": f"{prefix}.k_proj.weight",
        "v": f"{prefix}.v_proj.weight",
        "o": f"{prefix}.o_proj.weight",
    }
    require_shape(hf_state_dict, keys["q"], (query_hidden_size, hidden_size))
    require_shape(hf_state_dict, keys["k"], (kv_hidden_size, hidden_size))
    require_shape(hf_state_dict, keys["v"], (kv_hidden_size, hidden_size))
    require_shape(hf_state_dict, keys["o"], (hidden_size, query_hidden_size))

    bias_keys = {name: key[: -len(".weight")] + ".bias" for name, key in keys.items()}
    qkv_bias_presence = [bias_keys[name] in hf_state_dict for name in ("q", "k", "v")]
    if any(qkv_bias_presence) and not all(qkv_bias_presence):
        raise ValueError(
            "Q, K, and V projection biases must either all be present or all be absent"
        )
    if all(qkv_bias_presence):
        require_shape(hf_state_dict, bias_keys["q"], (query_hidden_size,))
        require_shape(hf_state_dict, bias_keys["k"], (kv_hidden_size,))
        require_shape(hf_state_dict, bias_keys["v"], (kv_hidden_size,))
    if bias_keys["o"] in hf_state_dict:
        require_shape(hf_state_dict, bias_keys["o"], (hidden_size,))

    projections = {name: hf_state_dict[key] for name, key in keys.items()}
    projection_biases = (
        {name: hf_state_dict[bias_keys[name]] for name in ("q", "k", "v")}
        if all(qkv_bias_presence)
        else None
    )
    output_bias = hf_state_dict.get(bias_keys["o"])
    return projections, projection_biases, output_bias


def convert_model(hf_state_dict, hf_config, tp_ranks):
    conv_state_dicts = [{} for _ in range(tp_ranks)]
    num_q_heads, num_kv_heads, head_dim = get_attention_dimensions(hf_config)
    validate_attention_tp(num_q_heads, num_kv_heads, tp_ranks)
    # get embeddings...
    for i, chunk in enumerate(
        torch.chunk(hf_state_dict["model.embed_tokens.weight"], tp_ranks, dim=0)
    ):
        conv_state_dicts[i][
            "sequential.0.word_embeddings.weight"
        ] = chunk.clone().detach()
    print(
        "model.embed_tokens.weight",
        hf_state_dict["model.embed_tokens.weight"].shape,
        "sequential.0.word_embeddings.weight",
        conv_state_dicts[0]["sequential.0.word_embeddings.weight"].shape,
    )
    # Get config data...
    # do layers...
    for layer_num in tqdm.tqdm(range(hf_config.num_hidden_layers)):
        # --- attention ---
        projections, projection_biases, output_bias = get_attention_projections(
            hf_state_dict, hf_config, layer_num, tp_ranks
        )
        # Output first since it's a simple row parallel...
        output_chunks = torch.split(
            projections["o"],
            num_q_heads // tp_ranks * head_dim,
            dim=1,
        )
        for i, chunk in enumerate(output_chunks):
            conv_state_dicts[i][
                f"sequential.{layer_num+2}.attention.dense.weight"
            ] = chunk.clone().detach()
            if output_bias is not None:
                conv_state_dicts[i][
                    f"sequential.{layer_num+2}.attention.dense.bias"
                ] = output_bias.clone().detach()
        print(
            f"model.layers.{layer_num}.self_attn.o_proj.weight",
            hf_state_dict[f"model.layers.{layer_num}.self_attn.o_proj.weight"].shape,
            f"sequential.{layer_num+2}.attention.dense.weight",
            conv_state_dicts[0][
                f"sequential.{layer_num+2}.attention.dense.weight"
            ].shape,
        )
        # Now for attention...
        # Split into heads...
        q = projections["q"]
        k = projections["k"]
        v = projections["v"]

        # Chunk for tensor parallelism...
        q_chunks = torch.split(q, num_q_heads // tp_ranks * head_dim, dim=0)
        k_chunks = torch.split(k, num_kv_heads // tp_ranks * head_dim, dim=0)
        v_chunks = torch.split(v, num_kv_heads // tp_ranks * head_dim, dim=0)
        for i, q_chunk, k_chunk, v_chunk in zip(
            range(tp_ranks),
            q_chunks,
            k_chunks,
            v_chunks,
        ):
            # The GQA code simply expects concatenated q,k,v weights for each tp partition
            conv_state_dicts[i][
                f"sequential.{layer_num+2}.attention.query_key_value.weight"
            ] = (torch.cat([q_chunk, k_chunk, v_chunk], dim=0).clone().detach())
            if projection_biases is not None:
                q_bias = torch.split(
                    projection_biases["q"],
                    num_q_heads // tp_ranks * head_dim,
                )[i]
                k_bias = torch.split(
                    projection_biases["k"],
                    num_kv_heads // tp_ranks * head_dim,
                )[i]
                v_bias = torch.split(
                    projection_biases["v"],
                    num_kv_heads // tp_ranks * head_dim,
                )[i]
                conv_state_dicts[i][
                    f"sequential.{layer_num+2}.attention.query_key_value.bias"
                ] = (torch.cat([q_bias, k_bias, v_bias], dim=0).clone().detach())
        print(
            f"model.layers.{layer_num}.self_attn.(q/k/v)_proj.weight",
            hf_state_dict[f"model.layers.{layer_num}.self_attn.q_proj.weight"].shape,
            hf_state_dict[f"model.layers.{layer_num}.self_attn.k_proj.weight"].shape,
            hf_state_dict[f"model.layers.{layer_num}.self_attn.v_proj.weight"].shape,
            f"sequential.{layer_num+2}.attention.query_key_value.weight",
            conv_state_dicts[0][
                f"sequential.{layer_num+2}.attention.query_key_value.weight"
            ].shape,
        )
        # --- mlp ---
        # Do SwiGLU weights...
        # w1...
        for i, (w1, w3) in enumerate(
            zip(
                torch.chunk(
                    hf_state_dict[f"model.layers.{layer_num}.mlp.gate_proj.weight"],
                    tp_ranks,
                    dim=0,
                ),
                torch.chunk(
                    hf_state_dict[f"model.layers.{layer_num}.mlp.up_proj.weight"],
                    tp_ranks,
                    dim=0,
                ),
            )
        ):
            conv_state_dicts[i][f"sequential.{layer_num+2}.mlp.linear1.weight"] = (
                torch.cat([w3.clone().detach(), w1.clone().detach()], dim=0)
            )
        print(
            f"model.layers.{layer_num}.mlp.gate_proj.weight",
            hf_state_dict[f"model.layers.{layer_num}.mlp.gate_proj.weight"].shape,
            f"model.layers.{layer_num}.mlp.up_proj.weight",
            hf_state_dict[f"model.layers.{layer_num}.mlp.up_proj.weight"].shape,
            f"sequential.{layer_num+2}.mlp.w3.weight",
            conv_state_dicts[0][f"sequential.{layer_num+2}.mlp.linear1.weight"].shape,
        )
        # w2 (output)...
        for i, chunk in enumerate(
            torch.chunk(
                hf_state_dict[f"model.layers.{layer_num}.mlp.down_proj.weight"],
                tp_ranks,
                dim=1,
            )
        ):
            conv_state_dicts[i][
                f"sequential.{layer_num+2}.mlp.linear2.weight"
            ] = chunk.clone().detach()
        print(
            f"model.layers.{layer_num}.mlp.down_proj.weight",
            hf_state_dict[f"model.layers.{layer_num}.mlp.down_proj.weight"].shape,
            f"sequential.{layer_num+2}.mlp.linear2.weight",
            conv_state_dicts[0][f"sequential.{layer_num+2}.mlp.linear2.weight"].shape,
        )
        # --- norm ---
        for i in range(tp_ranks):
            conv_state_dicts[i][f"sequential.{layer_num+2}.input_layernorm.scale"] = (
                hf_state_dict[f"model.layers.{layer_num}.input_layernorm.weight"]
                .clone()
                .detach()
            )
            conv_state_dicts[i][
                f"sequential.{layer_num+2}.post_attention_layernorm.scale"
            ] = (
                hf_state_dict[
                    f"model.layers.{layer_num}.post_attention_layernorm.weight"
                ]
                .clone()
                .detach()
            )

    # Get final ln/linear....
    index = hf_config.num_hidden_layers + 3
    for i in range(tp_ranks):
        conv_state_dicts[i][f"sequential.{index}.norm.scale"] = (
            hf_state_dict["model.norm.weight"].clone().detach()
        )
    index += 1
    # do output...
    for i, chunk in enumerate(
        torch.chunk(hf_state_dict["lm_head.weight"], tp_ranks, dim=0)
    ):
        conv_state_dicts[i][
            f"sequential.{index}.final_linear.weight"
        ] = chunk.clone().detach()
    print(
        "lm_head.weight",
        hf_state_dict["lm_head.weight"].shape,
        f"sequential.{index}.final_linear.weight",
        conv_state_dicts[0][f"sequential.{index}.final_linear.weight"].shape,
    )
    return conv_state_dicts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tp", type=int, default=1, help="Number of tensor parallelism ranks"
    )
    parser.add_argument(
        "--pp", type=int, default=0, help="Number of pipeline parallelism stages"
    )
    parser.add_argument("--model", type=str, default="gpt2", help="HF model name")
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to save model"
    )
    args = parser.parse_args()
    assert args.pp == 0, "Pipeline parallelism not supported yet"
    tokenizer = AutoTokenizer.from_pretrained(args.model).save_pretrained(
        args.model_path + "/tokenizer"
    )
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
    state_dict = model.state_dict()
    for key in state_dict.keys():
        print(key, state_dict[key].shape)
    os.makedirs(args.model_path, exist_ok=True)
    # Setup model directory...
    os.makedirs(f"{args.model_path}/0", exist_ok=True)
    # Save the latest file so neox can figure out where to grab the weights...
    with open(f"{args.model_path}/latest", "w") as f:
        f.write("0")
    # Convert the model...
    tp_state_dicts = convert_model(state_dict, model.model.config, args.tp)
    for i in range(args.tp):
        torch.save(
            {
                "dp_world_size": 1,
                "mp_world_size": args.tp,
                "optimizer": {},
                "global_steps": 1,
                "skipped_steps": 1,
                "iteration": 1,
                "module": tp_state_dicts[i],
            },
            f"{args.model_path}/0/mp_rank_{i:02d}_model_states.pt",
        )
