# Copyright (c) 2023, EleutherAI
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

import os
import sys

import yaml
import argparse
from tqdm import tqdm
from typing import List

import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
try:
    from transformers import LlamaConfig, LlamaForCausalLM
except ImportError:
    print("LLamaForCausalLM could not be imported. Please update your `transformers` installation and try again.")
    raise Exception

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from megatron.tokenizer import build_tokenizer


"""
A script for converting saved NeoX Checkpoints to Huggingface (HF) compatible GPT-NeoX type models.

Note that this script does not support all NeoX features.
Please investigate carefully whether your model is compatible with all architectures supported by the GPTNeoXForCausalLM class in HF.

(e.g. position embeddings such as AliBi may not be supported by Huggingface's GPT-NeoX architecture.
"""

def load_partitions(input_checkpoint_path, mp_partitions) -> List[torch.Tensor]:
    """Returns a list containing all states from a model (across MP partitions)"""

    loaded_tp_ranks = [
        torch.load(
            os.path.join(
                input_checkpoint_path,
                f"mp_rank_{i:02}_model_states.pt",
            ),
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        for i in range(mp_partitions)
    ]

    return loaded_tp_ranks


def get_state(
    state_dicts: List[torch.Tensor],
    key: str,
    layer_idx: int,
) -> torch.Tensor:
    """Accesses all MP partitions of a given layer/weight's state."""
    # main DeepSpeed saves each MP partition
    key = f"sequential.{layer_idx}.{key}"

    return [state_dict["module"][key] for state_dict in state_dicts]


def get_key(loaded_config, key, default=None):
    """
    Search for a given key in a NeoX yaml. normalizes underscores -> hyphens
    """
    key = key.replace("_", "-")
    try:
        return loaded_config[key]
    except KeyError:
        key = key.replace("-", "_")
        try:
            return loaded_config[key]
        except KeyError:
            return default


def create_config(neox_config):
    """take in a loaded yaml from NeoX and assign relevant values to HF config.
    Returns: transformers.AutoConfig() object
    """

    def gated_size(hidden_dim):
        # takes in a hidden dim and calculates intermediate dim of a LLaMAParallelMLP.
        # hidden-size * 8 / 3 , rounded up to nearest multiple of 256
        ff_dim = int(2 * hidden_dim * 4 / 3)
        ff_dim = 256 * ((ff_dim + 256 - 1) // 256)
        return ff_dim

    class TokenizerArgs:
        # kinda hacky.
        # this is to get something with the same interface as is used in build_tokenizer()
        # without diving into loading a neox_args object or using argparse etc.
        def __init__(self, neox_config):
            self.make_vocab_size_divisible_by = get_key(
                neox_config, "make-vocab-size-divisible-by", default=128
            )
            self.model_parallel_size = get_key(neox_config, "model-parallel-size")
            self.vocab_file = get_key(neox_config, "vocab-file")
            self.merge_file = get_key(neox_config, "merge-file")
            self.tokenizer_type = get_key(neox_config, "tokenizer-type")
            self.rank = 0

    args = TokenizerArgs(neox_config)
    
    tokenizer = build_tokenizer(args)
    try:  # GPT2TokenizerFast raises NotImplementedError
        pad_token = tokenizer.pad
    except:
        pad_token = (
            1  # pad defaulting to 1. follows convention from GPT-NeoX-20b tokenizer
        ) 

    # TODO: change the default value here based on discussion regarding `gpt_j_tied` config parameter's default
    use_tied_lns = get_key(neox_config, "gpt-j-tied", False)

    if use_tied_lns:
        raise NotImplementedError(
            """ERROR: Huggingface Transformers does not yet support a single shared layernorm
                per transformer block for GPT-NeoX models trained  w/ GPT-J parallel residuals.
                See https://github.com/EleutherAI/gpt-neox/pull/481 for further details."""
        )

    # set all config values.
    if not get_key(neox_config, "mlp-type") == "llama":
        hf_config = GPTNeoXConfig(
            vocab_size=args.padded_vocab_size,
            hidden_size=get_key(neox_config, "hidden-size"),
            num_hidden_layers=get_key(neox_config, "num-layers"),
            num_attention_heads=get_key(neox_config, "num-attention-heads"),
            intermediate_size=(get_key(neox_config, "hidden-size") * 4),
            hidden_act=get_key(neox_config, "activation", default="gelu"),
            rotary_pct=get_key(neox_config, "rotary-pct", default=1.0),
            rotary_emb_base=get_key(neox_config, "rotary-emb-base", default=10000),
            max_position_embeddings=get_key(neox_config, "max-position-embeddings"),
            initializer_range=get_key(neox_config, "init-method-std", 0.02),
            layer_norm_eps=get_key(neox_config, "layernorm-epsilon", 1e-5),
            use_cache=True,
            bos_token_id=tokenizer.eod,
            eos_token_id=tokenizer.eod,
            tie_word_embeddings=(not get_key(neox_config, "no-weight-tying", False)),
            use_parallel_residual=get_key(neox_config, "gpt-j-residual", False),
        )
    else:
        hf_config = LlamaConfig(
            vocab_size=args.padded_vocab_size,
            hidden_size=get_key(neox_config, "hidden-size"),
            num_hidden_layers=get_key(neox_config, "num-layers"),
            num_attention_heads=get_key(neox_config, "num-attention-heads"),
            num_key_value_heads=get_key(neox_config, "num-kv-heads"),
            intermediate_size=gated_size(get_key(neox_config, "hidden-size")) ,
            hidden_act=get_key(neox_config, "activation", default="silu"),
            max_position_embeddings=get_key(neox_config, "max-position-embeddings"),
            initializer_range=get_key(neox_config, "init-method-std", 0.02),
            rms_norm_eps=get_key(neox_config, "rms-norm-epsilon", 1.0e-6),
            use_cache=True,
            rope_theta=get_key(neox_config, "rotary_emb_base", 10000.0),
            tie_word_embeddings=(not get_key(neox_config, "no-weight-tying", False)),
        )

    return hf_config


def convert(input_checkpoint_path, loaded_config, output_checkpoint_path):
    """convert a NeoX checkpoint to a HF model format.
    should perform model-parallel merging correctly
    but only supports features allowed by HF GPT-NeoX implementation (e.g. rotary embeddings)
    """

    hf_config = LlamaConfig() #GPTNeoXConfig()

    hf_config = create_config(loaded_config)

    hf_model = LlamaForCausalLM(hf_config) #GPTNeoXForCausalLM(hf_config)

    # save model in fp16/bf16 if Deepspeed fp16 or bf16 mixed precision was used in config, else 32 bit weights
    fp16 = get_key(loaded_config, "fp16")
    if fp16:
        try:
            # this conditional is quite messy because there were a number of ways to specify bf16 or fp16 training
            # in DeeperSpeed v1.0 .
            if (fp16.get("fp16", None) or fp16["enabled"]) and not (fp16.get("type", None) == "bfloat16"):
                hf_model.half()
                print("Saving weights in fp16 precision...")
            elif fp16.get("type", None) == "bfloat16":
                hf_model.to(dtype=torch.bfloat16)
                print("Saving weights in bf16 precision...")
        except:
            print("Model not trained in fp16 / bf16 mixed precision, saving weights in fp32...")
    
    mp_partitions = get_key(loaded_config, "model-parallel-size")

    ### Embedding layer ###
    loaded_tp_ranks = load_partitions(input_checkpoint_path, mp_partitions)
    
    hf_model.model.embed_tokens.load_state_dict(
        {
            "weight": torch.cat(
                get_state(loaded_tp_ranks, "word_embeddings.weight", 0), dim=0
            )
        },
        strict=True
    )

    assert (
        hf_config.vocab_size == hf_model.model.embed_tokens.weight.shape[0]
    ), f"ERROR: calculated vocab size {hf_config.vocab_size} != embed param size {hf_model.gpt_neox.embed_in.shape[0]}"
    ### End Embedding Layer ###

    for layer_i in tqdm(range(get_key(loaded_config, "num-layers"))):

        # get layer from hf model
        hf_layer = hf_model.model.layers[layer_i]

        # + 2 bc of embed layer and a dummy _pre_transformer_block
        #loaded_tp_ranks = load_partitions(
        #    input_checkpoint_path, mp_partitions, layer_i + 2
        #)

        state_dict = {}
        # RowParallelLinear
        for key, hf_key in {
            "attention.dense.weight": "self_attn.o_proj.weight",
            # "attention.dense.weight": "attention.dense.weight",
            #"mlp.dense_4h_to_h.weight": "mlp.dense_4h_to_h.weight",
            "mlp.w2.weight": "mlp.down_proj.weight",
        }.items():
            #if key in loaded_tp_ranks[0].keys():
            state_dict[hf_key] = torch.cat(get_state(loaded_tp_ranks, key, layer_i + 2), dim=1)
        
        # average layernorm stats over mp ranks
        for key, hf_key in {
            #"input_layernorm.weight": "input_layernorm.weight",
            #"input_layernorm.bias": "input_layernorm.bias",
            #"post_attention_layernorm.weight": "post_attention_layernorm.weight",
            #"post_attention_layernorm.bias": "post_attention_layernorm.bias",
            "input_layernorm.scale": "input_layernorm.weight",
            "post_attention_layernorm.scale": "post_attention_layernorm.weight",
        }.items():
            #if key in loaded_tp_ranks[0].keys():
            state_dict[hf_key] = (sum(get_state(loaded_tp_ranks, key, layer_i + 2))) / len(
                loaded_tp_ranks
            )

        # LinearWithTPMerge
        # (ColumnParallelLinear)
        for key, hf_key in {
            #"mlp.dense_h_to_4h.weight": "mlp.dense_h_to_4h.weight",
            #"mlp.dense_h_to_4h.bias": "mlp.dense_h_to_4h.bias",
            "mlp.w1.weight": "mlp.gate_proj.weight",
            # "mlp.w1.bias": "mlp.gate_proj.bias",
            "mlp.w3.weight": "mlp.up_proj.weight",
            # "mlp.w3.bias": mlp.w3.bias",

            # "attention.query_key_value.weight",
            # "attention.query_key_value.bias",
        }.items():
            #if key in loaded_tp_ranks[0].keys():
            state_dict[hf_key] = torch.cat(get_state(loaded_tp_ranks, key, layer_i + 2), dim=0)

        # LinearWithTPSplitBias
        # (RowParallelLinear)
        #for key in [
            # "mlp.w2.bias", -> no bias
            # "mlp.dense_4h_to_h.bias",
            # "attention.dense.bias": "self_attn.o_proj.bias", no bias
        #]:
        #    if key in loaded_tp_ranks[0].keys():
        #        state_dict[key] = sum(get_state(loaded_tp_ranks, key, layer_i + 2))

        # Just take one
        state_dict["self_attn.rotary_emb.inv_freq"] = get_state(loaded_tp_ranks, "attention.rotary_emb.inv_freq", layer_i + 2)[0]
        # state_dict["attention.bias"] = hf_layer.state_dict()["attention.bias"]
        # state_dict["attention.masked_bias"] = hf_layer.state_dict()[
        #     "attention.masked_bias"
        # ]

        # for LLaMA: need to shard QKV proj.
        for key in [
            "attention.query_key_value.weight"
        ]:
            # merge across TP ranks
            sharded_qkv = torch.stack(get_state(loaded_tp_ranks, key, layer_i + 2), dim=0)
            print(sharded_qkv.shape) # -> should have shape [TP_SIZE, (hidden_size + 2 * kv_hidden_size) / TP_SIZE, hidden_size]

            sharded_qkv = sharded_qkv.view(
                    len(loaded_tp_ranks),
                    hf_config.num_attention_heads // len(loaded_tp_ranks),
                    int(hf_config.hidden_size // hf_config.num_attention_heads * (1+ 2 * hf_config.num_key_value_heads / hf_config.num_attention_heads)), 
                    hf_config.hidden_size
                    ) # is meant to convert to shape [TP_SIZE, NUM_Q_HEADS_PER_SHARD, dims_per_head * (1 + 2 * kv-to-q head ratio[=0.125]), hidden_size]
            print(sharded_qkv.shape)
            q, k, v = torch.split(
                    sharded_qkv, [
                        hf_config.hidden_size // hf_config.num_attention_heads,
                        int((hf_config.num_key_value_heads / hf_config.num_attention_heads) * hf_config.hidden_size // hf_config.num_attention_heads),
                        int((hf_config.num_key_value_heads / hf_config.num_attention_heads) * hf_config.hidden_size // hf_config.num_attention_heads),
                        ],
                    dim=2 #sharded_qkv.dim() - 2
                    ) # splits along the dims_per_head * (1 + 2 * kv-to-q head ratio[=0.125]) dim to get 3 tensors of sizes [TP_SIZE, NUM_Q_HEADS_PER_SHARD, dims_per_head, hidden_size] and 2 x [TP_SIZE, NUM_Q_HEADS_PER_SHARD, (dims_per_head / kv-to-q head ratio), hidden_size]
            print(q.shape, k.shape, v.shape)

            q, k, v = q.squeeze(dim=2), k.squeeze(dim=2), v.squeeze(dim=2)
            q = q.view(
                    hf_config.num_attention_heads,
                    hf_config.hidden_size // hf_config.num_attention_heads,
                    hf_config.hidden_size
                ).reshape(
                    hf_config.hidden_size, hf_config.hidden_size
                )
            k = k.reshape(
            hf_config.num_key_value_heads,
                    hf_config.hidden_size // hf_config.num_attention_heads,
                    hf_config.hidden_size
                ).reshape(
                    hf_config.hidden_size // hf_config.num_attention_heads * hf_config.num_key_value_heads, hf_config.hidden_size
                )
            v = v.reshape(
                    hf_config.num_key_value_heads,
                    hf_config.hidden_size // hf_config.num_attention_heads,
                    hf_config.hidden_size
                ).reshape(
                    hf_config.hidden_size // hf_config.num_attention_heads * hf_config.num_key_value_heads, hf_config.hidden_size
                )
            print(q.shape, k.shape, v.shape) 

            # raise ValueError
            # merged_qkv = torch.cat([t[key] for t in loaded_tp_ranks], dim=0)
            # chunk into separate Q, K, V projections and load
            # q, k, v = torch.chunk(merged_qkv,  3, dim=0)
            for hf_key, proj in zip(["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"], [q, k, v]):
                state_dict[hf_key] = proj.clone()
        print(state_dict.keys())
        # load state_dict into layer
        hf_layer.load_state_dict(state_dict, strict=True)

    # Load final layer norm
    # loaded_tp_ranks = load_partitions(
    #     input_checkpoint_path, mp_partitions, get_key(loaded_config, "num-layers") + 3
    # )

    norm_state_dict = {}
    for key, hf_key in {
        #"weight": "weight",
        "scale": "weight",
        #"bias": "bias",
    }.items():
        key = "norm." + key
        #if key in loaded_tp_ranks[0].keys():
        norm_state_dict[hf_key] = (sum(get_state(loaded_tp_ranks, key, get_key(loaded_config, "num-layers") + 3))) / len(loaded_tp_ranks)
    
    hf_model.model.norm.load_state_dict(norm_state_dict, strict=True)

    # del loaded_tp_ranks

    # Load output embedding
    # loaded_tp_ranks = load_partitions(
    #     input_checkpoint_path, mp_partitions, get_key(loaded_config, "num-layers") + 4
    # )

    hf_model.lm_head.load_state_dict(
        {
            "weight": torch.cat(
                get_state(loaded_tp_ranks, "final_linear.weight", get_key(loaded_config, "num-layers") + 4), dim=0
            ),
        },
        strict=True
    )

    del loaded_tp_ranks

    return hf_model


if __name__ == "__main__":

    # before running script:
    # `pip install --upgrade transformers`
    # `huggingface-cli login`
    #
    from huggingface_hub import create_repo, HfApi

    parser = argparse.ArgumentParser(
        description="Merge MP partitions and convert to HF Model."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to NeoX checkpoint, e.g. /path/to/model/global_step143000",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to config file for the input NeoX checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output dir, where to save the HF Model, tokenizer, and configs",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Set to true in order to upload to the HF Hub directly.",
    )
    args = parser.parse_args()

    with open(args.config_file) as f:
        loaded_config = yaml.full_load(f)

    hf_model = convert(args.input_dir, loaded_config, args.output_dir)

    hf_model.save_pretrained(args.output_dir)

    # save tokenizer to directory as well, for easy loading of model as a HF model
    tokenizer_type = get_key(loaded_config, "tokenizer-type")

    if tokenizer_type == "HFTokenizer":
        print(f"saving tokenizer from file {get_key(loaded_config, 'vocab-file')}")
        from transformers import PreTrainedTokenizerFast

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=get_key(loaded_config, "vocab-file")
        )
        print("loaded tokenizer: ", tokenizer)
        tokenizer.save_pretrained(args.output_dir)
        print("tokenizer saved!")

    if args.upload:
        repo_name = input("Provide a repository name for the HF Hub: ")
        create_repo(repo_name, repo_type="model", private=False, use_auth_token=True)

        api = HfApi()
        api.upload_folder(
            folder_path=args.output_dir,
            repo_id=repo_name,
            repo_type="model",
        )
