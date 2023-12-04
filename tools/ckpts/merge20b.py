# Copyright (c) 2021, EleutherAI
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
import yaml
import shutil
from tqdm import auto as tqdm_lib


VOCAB_SIZE = 50432
IGNORED_MODEL_STATE_KEYS = [
    "optimizer",
    "random_rng_state",
    "np_rng_state",
    "torch_rng_state",
    "cuda_rng_state",
    "rng_tracker_states",
]


def modify_config(input_config_path, output_config_path, output_dir):
    with open(input_config_path) as f:
        loaded_config = yaml.full_load(f)

    # replace model/pipeline parallel
    loaded_config["model_parallel_size"] = 1
    loaded_config["pipe_parallel_size"] = 1

    # replace load / save directories:
    loaded_config["load"] = output_dir
    loaded_config["save"] = output_dir

    # replace some other paths
    loaded_config["vocab_file"] = os.path.join(output_dir, "20B_tokenizer.json")
    loaded_config["log_dir"] = "./logs"

    # we need to make sure the resulting vocab size is correct
    # do this by modifying the 'make_vocab_size_divisible_by' argument to be
    # orig * (orig_mp / mp_out)
    loaded_config["make_vocab_size_divisible_by"] = VOCAB_SIZE

    # remove zero optimizer
    loaded_config["zero_optimization"]["stage"] = 0

    with open(output_config_path, "w") as f:
        yaml.dump(loaded_config, f)


def modify_model_states(input_model_state_path, output_model_state_path):
    model_state = torch.load(input_model_state_path)
    for key in IGNORED_MODEL_STATE_KEYS:
        del model_state[key]
    model_state["mp_world_size"] = 1
    model_state["dp_world_size"] = 1  # could make this configurable?
    model_state["args"]["model_parallel_size"] = 1
    model_state["args"]["make_vocab_size_divisible_by"] = VOCAB_SIZE
    torch.save(model_state, output_model_state_path)


def merge_model_weights(input_checkpoint_path, output_checkpoint_path):
    pbar = tqdm_lib.tqdm(total=47)

    # Load transformer layers
    for layer_i in range(44):
        pbar.set_description(f"Merging layer {layer_i}")
        filename_tp1 = f"layer_{layer_i + 2:02d}-model_00-model_states.pt"
        filename_tp2 = f"layer_{layer_i + 2:02d}-model_01-model_states.pt"
        loaded_tp1 = torch.load(os.path.join(input_checkpoint_path, filename_tp1))
        loaded_tp2 = torch.load(os.path.join(input_checkpoint_path, filename_tp2))
        # noinspection PyDictCreation
        merged = {}

        # RowParallelLinear
        merged["mlp.dense_4h_to_h.weight"] = torch.cat(
            [
                loaded_tp1["mlp.dense_4h_to_h.weight"],
                loaded_tp2["mlp.dense_4h_to_h.weight"],
            ],
            dim=1,
        )
        merged["attention.dense.weight"] = torch.cat(
            [
                loaded_tp1["attention.dense.weight"],
                loaded_tp2["attention.dense.weight"],
            ],
            dim=1,
        )
        merged["mlp.dense_4h_to_h.bias"] = (
            loaded_tp1["mlp.dense_4h_to_h.bias"] + loaded_tp2["mlp.dense_4h_to_h.bias"]
        )
        merged["attention.dense.bias"] = (
            loaded_tp1["attention.dense.bias"] + loaded_tp2["attention.dense.bias"]
        )

        # Layer Norms
        merged["input_layernorm.weight"] = (
            loaded_tp1["input_layernorm.weight"] + loaded_tp2["input_layernorm.weight"]
        ) / 2
        merged["input_layernorm.bias"] = (
            loaded_tp1["input_layernorm.bias"] + loaded_tp2["input_layernorm.bias"]
        ) / 2
        merged["post_attention_layernorm.weight"] = (
            loaded_tp1["post_attention_layernorm.weight"]
            + loaded_tp2["post_attention_layernorm.weight"]
        ) / 2
        merged["post_attention_layernorm.bias"] = (
            loaded_tp1["post_attention_layernorm.bias"]
            + loaded_tp2["post_attention_layernorm.bias"]
        ) / 2

        # ColumnParallelLinear
        merged["mlp.dense_h_to_4h.weight"] = torch.cat(
            [
                loaded_tp1["mlp.dense_h_to_4h.weight"],
                loaded_tp2["mlp.dense_h_to_4h.weight"],
            ],
            dim=0,
        )
        merged["mlp.dense_h_to_4h.bias"] = torch.cat(
            [
                loaded_tp1["mlp.dense_h_to_4h.bias"],
                loaded_tp2["mlp.dense_h_to_4h.bias"],
            ],
            dim=0,
        )
        merged["attention.query_key_value.weight"] = torch.cat(
            [
                loaded_tp1["attention.query_key_value.weight"],
                loaded_tp2["attention.query_key_value.weight"],
            ],
            dim=0,
        )
        merged["attention.query_key_value.bias"] = torch.cat(
            [
                loaded_tp1["attention.query_key_value.bias"],
                loaded_tp2["attention.query_key_value.bias"],
            ],
            dim=0,
        )

        # Just take one
        merged["attention.rotary_emb.inv_freq"] = loaded_tp1[
            "attention.rotary_emb.inv_freq"
        ]

        torch.save(merged, os.path.join(output_checkpoint_path, filename_tp1))
        del loaded_tp1
        del loaded_tp2
        pbar.update(1)

    # Load input embedding
    pbar.set_description(f"Merging input embedding")
    loaded_tp1 = torch.load(
        os.path.join(input_checkpoint_path, "layer_00-model_00-model_states.pt")
    )
    loaded_tp2 = torch.load(
        os.path.join(input_checkpoint_path, "layer_00-model_01-model_states.pt")
    )
    merged = {
        "word_embeddings.weight": torch.cat(
            [
                loaded_tp1["word_embeddings.weight"],
                loaded_tp2["word_embeddings.weight"],
            ],
            dim=0,
        )
    }
    torch.save(
        merged,
        os.path.join(output_checkpoint_path, "layer_00-model_00-model_states.pt"),
    )
    del loaded_tp1
    del loaded_tp2
    pbar.update(1)

    # Load final layer norm
    pbar.set_description(f"Merging final layer norm")
    loaded_tp1 = torch.load(
        os.path.join(input_checkpoint_path, "layer_47-model_00-model_states.pt")
    )
    loaded_tp2 = torch.load(
        os.path.join(input_checkpoint_path, "layer_47-model_01-model_states.pt")
    )
    merged = {
        "norm.weight": (loaded_tp1["norm.weight"] + loaded_tp2["norm.weight"]) / 2,
        "norm.bias": (loaded_tp1["norm.bias"] + loaded_tp2["norm.bias"]) / 2,
    }
    torch.save(
        merged,
        os.path.join(output_checkpoint_path, "layer_47-model_00-model_states.pt"),
    )
    del loaded_tp1
    del loaded_tp2
    pbar.update(1)

    # Load output embedding
    pbar.set_description(f"Merging output embedding")
    loaded_tp1 = torch.load(
        os.path.join(input_checkpoint_path, "layer_48-model_00-model_states.pt")
    )
    loaded_tp2 = torch.load(
        os.path.join(input_checkpoint_path, "layer_48-model_01-model_states.pt")
    )
    merged = {
        "final_linear.weight": torch.cat(
            [
                loaded_tp1["final_linear.weight"],
                loaded_tp2["final_linear.weight"],
            ],
            dim=0,
        ),
    }
    torch.save(
        merged,
        os.path.join(output_checkpoint_path, "layer_48-model_00-model_states.pt"),
    )
    del loaded_tp1
    del loaded_tp2
    pbar.update(1)
    pbar.set_description("Done.")


def merge(input_dir, output_dir):
    input_checkpoint_path = os.path.join(input_dir, "global_step150000")
    output_checkpoint_path = os.path.join(output_dir, "global_step150000")
    os.makedirs(output_checkpoint_path, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "configs"), exist_ok=True)
    for i in range(8):
        modify_model_states(
            input_model_state_path=os.path.join(
                input_checkpoint_path, f"mp_rank_{i:02d}_model_states.pt"
            ),
            output_model_state_path=os.path.join(
                output_checkpoint_path, f"mp_rank_{i:02d}_model_states.pt"
            ),
        )
    modify_config(
        input_config_path=os.path.join(input_dir, "configs", "20B.yml"),
        output_config_path=os.path.join(output_dir, "configs", "20B.yml"),
        output_dir=output_dir,
    )
    merge_model_weights(
        input_checkpoint_path=input_checkpoint_path,
        output_checkpoint_path=output_checkpoint_path,
    )
    shutil.copyfile(
        os.path.join(input_dir, "20B_tokenizer.json"),
        os.path.join(output_dir, "20B_tokenizer.json"),
    )
    with open(os.path.join(output_dir, "latest"), "w") as f:
        f.write("global_step150000")


def main():
    parser = argparse.ArgumentParser(description="Merge 20B checkpoint.")
    parser.add_argument(
        "--input_dir",
        type=str,
        help='Checkpoint dir, which should contain (e.g. a folder named "global_step150000")',
    )
    parser.add_argument(
        "--output_dir", type=str, help="Output dir, to save the 1-GPU weights configs"
    )
    args = parser.parse_args()
    merge(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
