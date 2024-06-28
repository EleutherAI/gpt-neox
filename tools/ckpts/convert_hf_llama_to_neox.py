import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import tqdm


def convert_model(hf_state_dict, hf_config, tp_ranks):
    conv_state_dicts = [{} for _ in range(tp_ranks)]
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
    num_kv_heads = hf_config.num_key_value_heads
    num_q_heads = hf_config.num_attention_heads
    head_dim = hf_config.hidden_size // num_q_heads
    # do layers...
    for layer_num in tqdm.tqdm(range(model.model.config.num_hidden_layers)):
        # --- attention ---
        # Output first since it's a simple row parallel...
        for i, chunk in enumerate(
            torch.chunk(
                hf_state_dict[f"model.layers.{layer_num}.self_attn.o_proj.weight"],
                tp_ranks,
                dim=1,
            )
        ):
            conv_state_dicts[i][
                f"sequential.{layer_num+2}.attention.dense.weight"
            ] = chunk.clone().detach()
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
        q = hf_state_dict[f"model.layers.{layer_num}.self_attn.q_proj.weight"]
        k = hf_state_dict[f"model.layers.{layer_num}.self_attn.k_proj.weight"]
        v = hf_state_dict[f"model.layers.{layer_num}.self_attn.v_proj.weight"]
        # The GQA code splits the heads by the num_q_heads so we also do that
        # here to ensure it matches...
        q = q.view(num_q_heads, -1, q.shape[-1])
        k = k.view(num_q_heads, -1, q.shape[-1])
        v = v.view(num_q_heads, -1, q.shape[-1])
        # Chunk for tensor parallelism...
        for i, q_chunk, k_chunk, v_chunk in zip(
            range(tp_ranks),
            torch.chunk(q, tp_ranks, dim=0),
            torch.chunk(k, tp_ranks, dim=0),
            torch.chunk(v, tp_ranks, dim=0),
        ):
            # Need to join the heads across q, k, v...
            conv_state_dicts[i][
                f"sequential.{layer_num+2}.attention.query_key_value.weight"
            ] = (
                torch.cat([q_chunk, k_chunk, v_chunk], dim=1)
                .view(-1, q.shape[-1])
                .clone()
                .detach()
            )
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
        for i, chunk in enumerate(
            torch.chunk(
                hf_state_dict[f"model.layers.{layer_num}.mlp.gate_proj.weight"],
                tp_ranks,
                dim=0,
            )
        ):
            conv_state_dicts[i][
                f"sequential.{layer_num+2}.mlp.w1.weight"
            ] = chunk.clone().detach()
        print(
            f"model.layers.{layer_num}.mlp.gate_proj.weight",
            hf_state_dict[f"model.layers.{layer_num}.mlp.gate_proj.weight"].shape,
            f"sequential.{layer_num+2}.mlp.w1.weight",
            conv_state_dicts[0][f"sequential.{layer_num+2}.mlp.w1.weight"].shape,
        )
        # w3...
        for i, chunk in enumerate(
            torch.chunk(
                hf_state_dict[f"model.layers.{layer_num}.mlp.up_proj.weight"],
                tp_ranks,
                dim=0,
            )
        ):
            conv_state_dicts[i][
                f"sequential.{layer_num+2}.mlp.w3.weight"
            ] = chunk.clone().detach()
        print(
            f"model.layers.{layer_num}.mlp.up_proj.weight",
            hf_state_dict[f"model.layers.{layer_num}.mlp.up_proj.weight"].shape,
            f"sequential.{layer_num+2}.mlp.w3.weight",
            conv_state_dicts[0][f"sequential.{layer_num+2}.mlp.w3.weight"].shape,
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
                f"sequential.{layer_num+2}.mlp.w2.weight"
            ] = chunk.clone().detach()
        print(
            f"model.layers.{layer_num}.mlp.down_proj.weight",
            hf_state_dict[f"model.layers.{layer_num}.mlp.down_proj.weight"].shape,
            f"sequential.{layer_num+2}.mlp.w2.weight",
            conv_state_dicts[0][f"sequential.{layer_num+2}.mlp.w2.weight"].shape,
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
    index = model.model.config.num_hidden_layers + 3
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
