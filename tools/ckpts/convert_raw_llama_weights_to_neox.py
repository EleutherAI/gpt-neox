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

import argparse
import os
import torch
import json
import math
import tqdm.auto as tqdm


INTERMEDIATE_SIZE_MAP = {
    "7B": 11008,
    "13B": 13824,
    "30B": 17920,
    "65B": 22016,
}
NUM_SHARDS = {
    "7B": 1,
    "13B": 2,
    "30B": 4,
    "65B": 8,
}


def compute_intermediate_size(n):
    return int(math.ceil(n * 8 / 3) + 255) // 256 * 256


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def write_file(text, path):
    with open(path, "w") as f:
        f.write(text)


def convert_model_pipeline(
    output_base_path, input_base_path, model_size: str, num_output_shards: int
):
    assert model_size in NUM_SHARDS

    model_path = os.path.join(output_base_path, "global_step0")
    os.makedirs(model_path, exist_ok=True)
    write_file("global_step0", os.path.join(output_base_path, "latest"))

    params = read_json(os.path.join(input_base_path, "params.json"))
    num_input_shards = NUM_SHARDS[model_size]
    num_layers = params["n_layers"]
    num_heads = params["n_heads"]
    num_heads_per_input_shard = num_heads // num_input_shards
    num_heads_per_output_shard = num_heads // num_output_shards
    hidden_size = params["dim"]
    dims_per_head = hidden_size // num_heads
    # base = 10000.0
    # inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))

    def permute_rotary(w):
        assert w.shape == (num_heads, dims_per_head, hidden_size)
        return (
            w.view(num_heads, dims_per_head // 2, 2, hidden_size)
            .transpose(1, 2)
            .reshape(num_heads, dims_per_head, hidden_size)
        )

    pbar = tqdm.tqdm(total=num_input_shards + num_layers + 3)

    pbar.set_description(f"Loading shard")
    loaded = []
    for i in range(num_input_shards):
        loaded.append(
            torch.load(
                os.path.join(input_base_path, f"consolidated.{i:02d}.pth"),
                map_location="cpu",
            )
        )
        pbar.set_description(f"Loaded shard {i}/{num_input_shards}")
        pbar.update(1)
    helper = Helper(
        loaded=loaded,
        model_path=model_path,
        num_output_shards=num_output_shards,
        model_size=model_size,
        pipeline_parallel=False,
    )

    sequential_cache = [{} for _ in range(num_output_shards)]

    # Embedding in
    embeddings_in = torch.cat(
        [
            loaded[rank]["tok_embeddings.weight"].cpu()
            for rank in range(num_input_shards)
        ],
        dim=1,
    )
    helper.save_shards(
        {"word_embeddings.weight": helper.shard(embeddings_in, dim=0)}, layer_i=0
    )
    helper.del_loaded("tok_embeddings.weight")
    pbar.set_description(f"Saved embeddings")
    pbar.update(1)

    # Norms
    helper.save_duplicates(
        {"norm.scale": loaded[0]["norm.weight"]}, layer_i=num_layers + 3
    )
    helper.del_loaded("norm.weight")
    pbar.set_description(f"Saved final norm")
    pbar.update(1)

    # Embedding out
    embeddings_out = torch.cat(
        [loaded[rank]["output.weight"].cpu() for rank in range(num_input_shards)], dim=0
    )
    helper.save_shards(
        {"final_linear.weight": helper.shard(embeddings_out, dim=0)},
        layer_i=num_layers + 4,
    )
    helper.del_loaded("output.weight")
    pbar.set_description(f"Saved out embeddings")
    pbar.update(1)

    # Layers
    if model_size == "7B":
        rope_freqs = loaded[0]["layers.0.attention.inner_attention.rope.freqs"]
        helper.del_loaded("layers.0.attention.inner_attention.rope.freqs")
    else:
        rope_freqs = loaded[0]["rope.freqs"]
        helper.del_loaded("rope.freqs")
    for layer_i in range(num_layers):

        # Linear
        attn_wo = helper.shard(
            torch.cat(
                [
                    loaded[rank][f"layers.{layer_i}.attention.wo.weight"]
                    for rank in range(num_input_shards)
                ],
                dim=1,
            ),
            dim=1,
        )
        mlp_w1 = helper.shard(
            torch.cat(
                [
                    loaded[rank][f"layers.{layer_i}.feed_forward.w1.weight"]
                    for rank in range(num_input_shards)
                ],
                dim=0,
            ),
            dim=0,
        )
        mlp_w2 = helper.shard(
            torch.cat(
                [
                    loaded[rank][f"layers.{layer_i}.feed_forward.w2.weight"]
                    for rank in range(num_input_shards)
                ],
                dim=1,
            ),
            dim=1,
        )
        mlp_w3 = helper.shard(
            torch.cat(
                [
                    loaded[rank][f"layers.{layer_i}.feed_forward.w3.weight"]
                    for rank in range(num_input_shards)
                ],
                dim=0,
            ),
            dim=0,
        )
        helper.del_loaded(f"layers.{layer_i}.attention.wo.weight")
        helper.del_loaded(f"layers.{layer_i}.feed_forward.w1.weight")
        helper.del_loaded(f"layers.{layer_i}.feed_forward.w2.weight")
        helper.del_loaded(f"layers.{layer_i}.feed_forward.w3.weight")

        # Attention
        w_q = permute_rotary(
            torch.cat(
                [
                    loaded[rank][f"layers.{layer_i}.attention.wq.weight"].view(
                        num_heads_per_input_shard, dims_per_head, hidden_size
                    )
                    for rank in range(num_input_shards)
                ],
                dim=0,
            )
        )
        w_k = permute_rotary(
            torch.cat(
                [
                    loaded[rank][f"layers.{layer_i}.attention.wk.weight"].view(
                        num_heads_per_input_shard, dims_per_head, hidden_size
                    )
                    for rank in range(num_input_shards)
                ],
                dim=0,
            )
        )
        w_v = torch.cat(
            [
                loaded[rank][f"layers.{layer_i}.attention.wv.weight"].view(
                    num_heads_per_input_shard, dims_per_head, hidden_size
                )
                for rank in range(num_input_shards)
            ],
            dim=0,
        )
        sharded_qkv = torch.stack(
            [
                helper.shard(
                    w_q, dim=0
                ),  # num_output_shards, num_heads_per_output_shard, dims_per_head, hidden_size
                helper.shard(w_k, dim=0),
                helper.shard(w_v, dim=0),
            ],
            dim=2,
        )  # num_output_shards, num_heads_per_output_shard, QKV=3, dims_per_head, hidden_size
        sharded_qkv = sharded_qkv.view(
            num_output_shards,
            num_heads_per_output_shard * 3 * dims_per_head,
            hidden_size,
        )
        helper.del_loaded(f"layers.{layer_i}.attention.wq.weight")
        helper.del_loaded(f"layers.{layer_i}.attention.wk.weight")
        helper.del_loaded(f"layers.{layer_i}.attention.wv.weight")

        # Duplicated
        input_layernorm = loaded[0][f"layers.{layer_i}.attention_norm.weight"]
        post_attention_layernorm = loaded[0][f"layers.{layer_i}.ffn_norm.weight"]
        helper.del_loaded(f"layers.{layer_i}.attention_norm.weight")
        helper.del_loaded(f"layers.{layer_i}.ffn_norm.weight")

        for out_rank in range(num_output_shards):
            helper.save(
                {
                    "attention.query_key_value.weight": sharded_qkv[out_rank],
                    # Sharded layers
                    "attention.dense.weight": attn_wo[out_rank].clone(),
                    "mlp.w1.weight": mlp_w1[out_rank].clone(),
                    "mlp.w2.weight": mlp_w2[out_rank].clone(),
                    "mlp.w3.weight": mlp_w3[out_rank].clone(),
                    # Duplicated layers
                    "input_layernorm.scale": input_layernorm,
                    "post_attention_layernorm.scale": post_attention_layernorm,
                    "attention.rotary_emb.inv_freq": rope_freqs,
                },
                layer_i=layer_i + 2,
                rank=out_rank,
            )

        pbar.set_description(f"Saved layer {layer_i} / {num_layers}")
        pbar.update(1)

    model_state = {
        "dp_world_size": 1,
        "mp_world_size": num_output_shards,
        "module": {},
        "optimizer": {},
        "global_steps": 1,
        "skipped_steps": 1,
        "iteration": 1,
    }
    for rank in range(num_output_shards):
        torch.save(
            model_state, os.path.join(model_path, f"mp_rank_{rank:02d}_model_states.pt")
        )
    pbar.set_description("Done.")


def convert_model_sequential(
    output_base_path, input_base_path, model_size: str, num_output_shards: int
):
    assert model_size in NUM_SHARDS

    model_path = os.path.join(output_base_path, "global_step0")
    os.makedirs(model_path, exist_ok=True)
    write_file("global_step0", os.path.join(output_base_path, "latest"))

    params = read_json(os.path.join(input_base_path, "params.json"))
    num_input_shards = NUM_SHARDS[model_size]
    num_layers = params["n_layers"]
    num_heads = params["n_heads"]
    num_heads_per_input_shard = num_heads // num_input_shards
    num_heads_per_output_shard = num_heads // num_output_shards
    hidden_size = params["dim"]
    dims_per_head = hidden_size // num_heads
    # base = 10000.0
    # inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))

    def permute_rotary(w):
        assert w.shape == (num_heads, dims_per_head, hidden_size)
        return (
            w.view(num_heads, dims_per_head // 2, 2, hidden_size)
            .transpose(1, 2)
            .reshape(num_heads, dims_per_head, hidden_size)
        )

    pbar = tqdm.tqdm(total=num_input_shards + num_output_shards)

    pbar.set_description(f"Loading shard")
    loaded = []
    for i in range(num_input_shards):
        loaded.append(
            torch.load(
                os.path.join(input_base_path, f"consolidated.{i:02d}.pth"),
                map_location="cpu",
            )
        )
        pbar.set_description(f"Loaded shard {i}/{num_input_shards}")
        pbar.update(1)
    helper = Helper(
        loaded=loaded,
        model_path=model_path,
        num_output_shards=num_output_shards,
        model_size=model_size,
        pipeline_parallel=False,
    )

    # Embedding in
    embeddings_in = torch.cat(
        [
            loaded[rank]["tok_embeddings.weight"].cpu()
            for rank in range(num_input_shards)
        ],
        dim=1,
    )
    helper.add_sequential_shard(
        {"word_embeddings.weight": helper.shard(embeddings_in, dim=0)}, layer_i=0
    )
    helper.del_loaded("tok_embeddings.weight")

    # Norms
    helper.add_sequential_duplicates(
        {"norm.scale": loaded[0]["norm.weight"]}, layer_i=num_layers + 3
    )
    helper.del_loaded("norm.weight")

    # Embedding out
    embeddings_out = torch.cat(
        [loaded[rank]["output.weight"].cpu() for rank in range(num_input_shards)], dim=0
    )
    helper.add_sequential_shard(
        {"final_linear.weight": helper.shard(embeddings_out, dim=0)},
        layer_i=num_layers + 4,
    )
    helper.del_loaded("output.weight")

    # Layers
    if model_size == "7B":
        rope_freqs = loaded[0]["layers.0.attention.inner_attention.rope.freqs"]
        helper.del_loaded("layers.0.attention.inner_attention.rope.freqs")
    else:
        rope_freqs = loaded[0]["rope.freqs"]
        helper.del_loaded("rope.freqs")
    for layer_i in range(num_layers):

        # Linear
        attn_wo = helper.shard(
            torch.cat(
                [
                    loaded[rank][f"layers.{layer_i}.attention.wo.weight"]
                    for rank in range(num_input_shards)
                ],
                dim=1,
            ),
            dim=1,
        )
        mlp_w1 = helper.shard(
            torch.cat(
                [
                    loaded[rank][f"layers.{layer_i}.feed_forward.w1.weight"]
                    for rank in range(num_input_shards)
                ],
                dim=0,
            ),
            dim=0,
        )
        mlp_w2 = helper.shard(
            torch.cat(
                [
                    loaded[rank][f"layers.{layer_i}.feed_forward.w2.weight"]
                    for rank in range(num_input_shards)
                ],
                dim=1,
            ),
            dim=1,
        )
        mlp_w3 = helper.shard(
            torch.cat(
                [
                    loaded[rank][f"layers.{layer_i}.feed_forward.w3.weight"]
                    for rank in range(num_input_shards)
                ],
                dim=0,
            ),
            dim=0,
        )
        helper.del_loaded(f"layers.{layer_i}.attention.wo.weight")
        helper.del_loaded(f"layers.{layer_i}.feed_forward.w1.weight")
        helper.del_loaded(f"layers.{layer_i}.feed_forward.w2.weight")
        helper.del_loaded(f"layers.{layer_i}.feed_forward.w3.weight")

        # Attention
        w_q = permute_rotary(
            torch.cat(
                [
                    loaded[rank][f"layers.{layer_i}.attention.wq.weight"].view(
                        num_heads_per_input_shard, dims_per_head, hidden_size
                    )
                    for rank in range(num_input_shards)
                ],
                dim=0,
            )
        )
        w_k = permute_rotary(
            torch.cat(
                [
                    loaded[rank][f"layers.{layer_i}.attention.wk.weight"].view(
                        num_heads_per_input_shard, dims_per_head, hidden_size
                    )
                    for rank in range(num_input_shards)
                ],
                dim=0,
            )
        )
        w_v = torch.cat(
            [
                loaded[rank][f"layers.{layer_i}.attention.wv.weight"].view(
                    num_heads_per_input_shard, dims_per_head, hidden_size
                )
                for rank in range(num_input_shards)
            ],
            dim=0,
        )
        sharded_qkv = torch.stack(
            [
                helper.shard(
                    w_q, dim=0
                ),  # num_output_shards, num_heads_per_output_shard, dims_per_head, hidden_size
                helper.shard(w_k, dim=0),
                helper.shard(w_v, dim=0),
            ],
            dim=2,
        )  # num_output_shards, num_heads_per_output_shard, QKV=3, dims_per_head, hidden_size
        sharded_qkv = sharded_qkv.view(
            num_output_shards,
            num_heads_per_output_shard * 3 * dims_per_head,
            hidden_size,
        )
        helper.del_loaded(f"layers.{layer_i}.attention.wq.weight")
        helper.del_loaded(f"layers.{layer_i}.attention.wk.weight")
        helper.del_loaded(f"layers.{layer_i}.attention.wv.weight")

        # Duplicated
        input_layernorm = loaded[0][f"layers.{layer_i}.attention_norm.weight"]
        post_attention_layernorm = loaded[0][f"layers.{layer_i}.ffn_norm.weight"]
        helper.del_loaded(f"layers.{layer_i}.attention_norm.weight")
        helper.del_loaded(f"layers.{layer_i}.ffn_norm.weight")

        for out_rank in range(num_output_shards):
            helper.add_sequential(
                {
                    "attention.query_key_value.weight": sharded_qkv[out_rank],
                    # Sharded layers
                    "attention.dense.weight": attn_wo[out_rank].clone(),
                    "mlp.w1.weight": mlp_w1[out_rank].clone(),
                    "mlp.w2.weight": mlp_w2[out_rank].clone(),
                    "mlp.w3.weight": mlp_w3[out_rank].clone(),
                    # Duplicated layers
                    "input_layernorm.scale": input_layernorm,
                    "post_attention_layernorm.scale": post_attention_layernorm,
                    "attention.rotary_emb.inv_freq": rope_freqs,
                },
                layer_i=layer_i + 2,
                rank=out_rank,
            )

    for rank in range(num_output_shards):
        model_state = {
            "dp_world_size": 1,
            "mp_world_size": num_output_shards,
            "module": helper.sequential_cache[rank],
            "optimizer": {},
            "global_steps": 1,
            "skipped_steps": 1,
            "iteration": 1,
        }
        torch.save(
            model_state, os.path.join(model_path, f"mp_rank_{rank:02d}_model_states.pt")
        )
        pbar.set_description(f"Saved shard {rank}")
        pbar.update(1)
    pbar.set_description("Done.")


class Helper:
    def __init__(
        self, loaded, model_size, num_output_shards, model_path, pipeline_parallel
    ):
        self.loaded = loaded
        self.model_size = model_size
        self.num_output_shards = num_output_shards
        self.model_path = model_path

        self.pipeline_parallel = pipeline_parallel
        self.sequential_cache = [{} for _ in range(num_output_shards)]

    def del_loaded(self, key: str):
        # Remove from memory as we go along
        for loaded_shared in self.loaded:
            del loaded_shared[key]

    def save_shards(self, dictionary, layer_i: int):
        for k, v in dictionary.items():
            assert v.shape[0] == self.num_output_shards
        for rank in range(self.num_output_shards):
            torch.save(
                {k: v[rank].clone() for k, v in dictionary.items()},
                self.save_path(layer_i=layer_i, rank=rank),
            )

    def save_duplicates(self, dictionary, layer_i: int):
        for rank in range(self.num_output_shards):
            torch.save(
                {k: v.clone() for k, v in dictionary.items()},
                self.save_path(layer_i=layer_i, rank=rank),
            )

    def save(self, obj, layer_i, rank):
        torch.save(obj, self.save_path(layer_i=layer_i + 2, rank=rank))

    def shard(self, x, dim):
        x_shape = list(x.shape)
        assert x_shape[dim] % self.num_output_shards == 0
        new_x_shape = (
            x_shape[:dim]
            + [self.num_output_shards, x_shape[dim] // self.num_output_shards]
            + x_shape[dim + 1 :]
        )
        x = x.view(*new_x_shape)
        return torch.movedim(x, 0, dim)

    def save_path(self, layer_i, rank):
        return os.path.join(
            self.model_path, f"layer_{layer_i:02d}-model_{rank:02d}-model_states.pt"
        )

    def add_sequential_shard(self, dictionary, layer_i):
        assert not self.pipeline_parallel
        for k, v in dictionary.items():
            for rank in range(self.num_output_shards):
                self.sequential_cache[rank][f"sequential.{layer_i}.{k}"] = v[
                    rank
                ].clone()

    def add_sequential_duplicates(self, dictionary, layer_i):
        assert not self.pipeline_parallel
        for k, v in dictionary.items():
            for rank in range(self.num_output_shards):
                self.sequential_cache[rank][f"sequential.{layer_i}.{k}"] = v.clone()

    def add_sequential(self, dictionary, layer_i, rank):
        assert not self.pipeline_parallel
        for k, v in dictionary.items():
            self.sequential_cache[rank][f"sequential.{layer_i}.{k}"] = v.clone()


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw LLaMA checkpoints to GPT-NeoX format."
    )
    parser.add_argument(
        "--input_dir",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--model_size",
        choices=["7B", "13B", "30B", "65B", "tokenizer_only"],
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write GPT-NeoX mode",
    )
    parser.add_argument(
        "--num_output_shards",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--pipeline_parallel",
        action="store_true",
        help="Only use if PP>1",
    )
    args = parser.parse_args()
    if args.pipeline_parallel:
        print("parallel")
        convert_model_pipeline(
            output_base_path=args.output_dir,
            input_base_path=os.path.join(args.input_dir, args.model_size),
            model_size=args.model_size,
            num_output_shards=args.num_output_shards,
        )
    else:
        print("sequential")
        convert_model_sequential(
            output_base_path=args.output_dir,
            input_base_path=os.path.join(args.input_dir, args.model_size),
            model_size=args.model_size,
            num_output_shards=args.num_output_shards,
        )


if __name__ == "__main__":
    main()
