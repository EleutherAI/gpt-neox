import argparse
import os

import torch
from transformers import GPTJForCausalLM, GPTJConfig
from oslo.torch.nn.parallel.tensor_parallel import TensorParallel
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn.parallel.utils import unwrap_parallel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", default="./startpoint/config.json")
    parser.add_argument("--checkpoint_path", type=str, default="./startpoint/model_6b.pt")
    parser.add_argument("--save_path", type=str, default="./checkpoints/6B")
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--make_vocab_size_divisible_by", type=int, default=128)
    return parser.parse_args()


def _resize_vocab_size(model, make_vocab_size_divisible_by):
    module = model.transformer.wte
    vocab_size, embedding_dim = module.weight.size()
    new_vocab_size = vocab_size

    while new_vocab_size % make_vocab_size_divisible_by != 0:
        new_vocab_size += 1
    
    if new_vocab_size != vocab_size:
        padding = torch.zeros(
            new_vocab_size - vocab_size,
            embedding_dim,
            dtype=module.weight.dtype,
        )
        new_embeddings = torch.cat(
            tensors=[module.weight.data, padding],
            dim=0,
        )
        module.weight.data = new_embeddings
        module.num_embeddings = new_vocab_size
    return model


def _resize_lm_head_size(model, make_vocab_size_divisible_by):
    module = model.lm_head
    out_features, in_features = module.weight.size()
    new_out_features = out_features

    while new_out_features % make_vocab_size_divisible_by != 0:
        new_out_features += 1
    
    if new_out_features != out_features:
        padding = torch.zeros(
            new_out_features - out_features,
            in_features,
            dtype=module.weight.dtype,
        )
        new_weight = torch.cat(
            tensors=[module.weight.data, padding],
            dim=0,
        )

        if hasattr(module, "bias") and module.bias is not None:
            padding = torch.zeros(
                new_out_features - out_features,
                dtype=module.weight.dtype,
            )
            new_bias = torch.cat(
                tensors=[module.bias.data, padding],
                dim=0,
            )
            module.bias.data = new_bias
        module.weight.data = new_weight
        module.out_features = new_out_features
    return model


def main():
    args = parse_args()
    if args.config_name:
        config = GPTJConfig.from_pretrained(args.config_name)
        model = GPTJForCausalLM(config)
        model.load_state_dict(torch.load(args.checkpoint_path))
    elif os.path.isdir(args.checkpoint_path):
        model = GPTJForCausalLM.from_pretrained(args.checkpoint_path)
        config = model.config
    else:
        raise ValueError("You should provide either config_name and checkpoint_path" 
                         " loadable (with torch.load) or checkpoint_path as directory" 
                         " (loadable with from_pretrained).")

    parallel_context = ParallelContext.from_torch(
        data_parallel_size=1,
        pipeline_parallel_size=1,
        tensor_parallel_size=args.tensor_parallel_size,
        tensor_parallel_mode=ParallelMode.TENSOR_1D,
    )

    os.makedirs(args.save_path, exist_ok=True)
    
    rank = parallel_context.get_local_rank(ParallelMode.TENSOR_1D)
    num_layers = config.n_layer
    rotary_dim = config.rotary_dim
    n_embd = config.n_embd
    
    model = _resize_vocab_size(model, args.make_vocab_size_divisible_by)
    model = _resize_lm_head_size(model, args.make_vocab_size_divisible_by)

    model = TensorParallel(model, parallel_context)
    model = unwrap_parallel(model)

    src_state_dict = model.state_dict()
    dtype = src_state_dict["transformer.wte.weight"].dtype

    # word embedding
    tgt_state_dict = {}
    tgt_state_dict["word_embeddings.weight"] = src_state_dict["transformer.wte.weight"]

    torch.save(tgt_state_dict, os.path.join(args.save_path, f"layer_00-model_{rank:02d}-model_states.pt"))

    for layer_idx in range(num_layers):
        tgt_state_dict = {}
        # input layernorm
        tgt_state_dict["input_layernorm.weight"] = src_state_dict[f"transformer.h.{layer_idx}.ln_1.weight"]
        tgt_state_dict["input_layernorm.bias"] = src_state_dict[f"transformer.h.{layer_idx}.ln_1.bias"]
        
        # attention query_key_value
        qkv_proj_weight = torch.cat([
            src_state_dict[f"transformer.h.{layer_idx}.attn.q_proj.weight"],
            src_state_dict[f"transformer.h.{layer_idx}.attn.k_proj.weight"],
            src_state_dict[f"transformer.h.{layer_idx}.attn.v_proj.weight"],
        ], dim=0)
        tgt_state_dict["attention.query_key_value.weight"] = qkv_proj_weight
        # Since there is no qkv bias in GPTJ, we set bias to 0.
        tgt_state_dict["attention.query_key_value.bias"] = torch.zeros(qkv_proj_weight.shape[0], dtype=dtype)

        # rotary embedding
        # There is no inv freq in GPTJ but it has deterministic values as follow.
        inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
        tgt_state_dict["attention.rotary_emb.inv_freq"] = inv_freq.type(dtype)

        # attention dense
        out_proj_weight = src_state_dict[f"transformer.h.{layer_idx}.attn.out_proj.weight"]
        tgt_state_dict["attention.dense.weight"] = out_proj_weight
        # Since there is no dense bias in GPTJ, we set bias to 0.
        tgt_state_dict["attention.dense.bias"] = torch.zeros(out_proj_weight.shape[0], dtype=dtype)

        # post layernorm
        # Since there is no post layernorm in GPTJ, we set weight to 1 and bias to 0.
        tgt_state_dict["post_attention_layernorm.weight"] = torch.ones(n_embd, dtype=dtype)
        tgt_state_dict["post_attention_layernorm.bias"] = torch.zeros(n_embd, dtype=dtype)

        # mlp
        tgt_state_dict["mlp.dense_h_to_4h.weight"] = src_state_dict[f"transformer.h.{layer_idx}.mlp.fc_in.weight"]
        tgt_state_dict["mlp.dense_h_to_4h.bias"] = src_state_dict[f"transformer.h.{layer_idx}.mlp.fc_in.bias"]
        tgt_state_dict["mlp.dense_4h_to_h.weight"] = src_state_dict[f"transformer.h.{layer_idx}.mlp.fc_out.weight"]
        tgt_state_dict["mlp.dense_4h_to_h.bias"] = src_state_dict[f"transformer.h.{layer_idx}.mlp.fc_out.bias"]
        
        torch.save(tgt_state_dict, os.path.join(args.save_path, f"layer_{layer_idx+2:02d}-model_{rank:02d}-model_states.pt"))
    
    # final norm
    tgt_state_dict = {}
    tgt_state_dict["norm.weight"] = src_state_dict["transformer.ln_f.weight"]
    tgt_state_dict["norm.bias"] = src_state_dict["transformer.ln_f.bias"]
    torch.save(tgt_state_dict, os.path.join(args.save_path, f"layer_{num_layers+3:02d}-model_{rank:02d}-model_states.pt"))

    # final linear
    tgt_state_dict = {}
    tgt_state_dict["final_linear.weight"] = src_state_dict["lm_head.weight"]
    torch.save(tgt_state_dict, os.path.join(args.save_path, f"layer_{num_layers+4:02d}-model_{rank:02d}-model_states.pt"))


if __name__ == "__main__":
    main()
