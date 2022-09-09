import argparse
import os
import torch
import yaml
import shutil
from tqdm import auto as tqdm_lib


IGNORED_MODEL_STATE_KEYS = [
    "optimizer",
    "random_rng_state",
    "np_rng_state",
    "torch_rng_state",
    "cuda_rng_state",
    "rng_tracker_states",
]


def modify_config(input_config_path, output_config_path, output_dir, vocab_size):
    with open(input_config_path) as f:
        loaded_config = yaml.full_load(f)

    # replace model/pipeline parallel
    loaded_config["model-parallel-size"] = 1
    loaded_config["pipe-parallel-size"] = 1

    # replace load / save directories:
    loaded_config["load"] = output_dir
    loaded_config["save"] = output_dir

    # replace some other paths
    loaded_config["vocab-file"] = os.path.join(output_dir, "tokenizer.json") # TODO: change variable
    loaded_config["log-dir"] = "./logs"

    # we need to make sure the resulting vocab size is correct
    # do this by modifying the 'make_vocab_size_divisible_by' argument to be
    # orig * (orig_mp / mp_out)
    loaded_config["make_vocab_size_divisible_by"] = vocab_size

    # remove zero optimizer
    loaded_config["zero_optimization"]["stage"] = 0

    with open(output_config_path, "w") as f:
        yaml.dump(loaded_config, f)


def modify_model_states(vocab_size, input_model_state_path, output_model_state_path):
    model_state = torch.load(input_model_state_path)
    for key in IGNORED_MODEL_STATE_KEYS:
        del model_state[key]
    model_state["mp_world_size"] = 1
    model_state["dp_world_size"] = 1  # could make this configurable?
    model_state["args"]["model_parallel_size"] = 1
    model_state["args"]["make_vocab_size_divisible_by"] = vocab_size
    torch.save(model_state, output_model_state_path)


def merge_model_weights(config, input_checkpoint_path, output_checkpoint_path):

    num_layers = config.get('num-layers', 44)
    mp_size = config.get('model-parallel-size', 1)

    pbar = tqdm_lib.tqdm(total=num_layers+3)

    # Load transformer layers
    for layer_i in range(num_layers): # num layers, TODO: change variable 
        pbar.set_description(f"Merging layer {layer_i}")

        tp_file_names = [
            f"layer_{layer_i + 2:02d}-model_{i:02d}-model_states.pt" 
            for i in range(mp_size)    
        ]
        loaded_tps = [
            torch.load(os.path.join(input_checkpoint_path, tpf))
            for tpf in tp_file_names    
        ]
        # loaded_tp1 = torch.load(os.path.join(input_checkpoint_path, filename_tp1))
        # loaded_tp2 = torch.load(os.path.join(input_checkpoint_path, filename_tp2))
        merged = {}
        merge_keys = {
            "row": [
                "mlp.dense_4h_to_h.weight",
                "attention.dense.weight",
                "mlp.dense_4h_to_h.bias",
                "attention.dense.bias"
            ],
            "layernorm": [
                "input_layernorm.weight",
                "input_layernorm.bias",
                "post_attention_layernorm.weight",
                "post_attention_layernorm.bias"
            ],
            "col": [
                "mlp.dense_h_to_4h.weight",
                "mlp.dense_h_to_4h.bias",
                "attention.query_key_value.weight",
                "attention.query_key_value.bias"
            ]
        }

        for k, model_keys in merge_keys.items():
            dim = None
            if k == 'row':
                dim = 1
            else:
                dim = 0

            for mk in model_keys:
                parallel_tensors = [tp[mk] for tp in loaded_tps]
                if dim is None:
                    merged[mk] = torch.sum(
                        torch.stack(parallel_tensors), dim=0
                    ) / mp_size
                elif mk.split('.')[-1] == 'weight':
                    merged[mk] = torch.cat(parallel_tensors, dim=dim)
                else:
                    merged[mk] = torch.sum(
                        torch.stack(parallel_tensors), dim=0
                    ) / mp_size

        # Just take one
        merged["attention.rotary_emb.inv_freq"] = loaded_tps[0]["attention.rotary_emb.inv_freq"]
        print(merged["attention.rotary_emb.inv_freq"].shape)

        torch.save(merged, os.path.join(output_checkpoint_path, tp_file_names[0]))
        del loaded_tps
        pbar.update(1)

    # Load input embedding
    pbar.set_description(f"Merging input embedding")

    loaded_embeddings = [
        torch.load(os.path.join(input_checkpoint_path, f"layer_00-model_{i:02d}-model_states.pt"))["word_embeddings.weight"]
        for i in range(mp_size)
    ]
    merged = {"word_embeddings.weight": torch.cat(loaded_embeddings, dim=0)}
    torch.save(merged, os.path.join(output_checkpoint_path, "layer_00-model_00-model_states.pt"))
    del loaded_embeddings
    pbar.update(1)

    # Load final layer norm
    pbar.set_description(f"Merging final layer norm")

    final_layer = num_layers + 3
    # TODO: change variable 47 => num_layers + 3
    loaded_tps = [
        torch.load(os.path.join(input_checkpoint_path, f"layer_{final_layer}-model_{i:02d}-model_states.pt")) 
        for i in range(mp_size)
    ]

    merged = {}
    for k in ["norm.weight", "norm.bias"]:
        merged[k] = torch.sum(torch.stack([tp[k] for tp in loaded_tps]), dim=0) / mp_size
    # TODO: change variable 47 => num_layers + 3
    torch.save(merged, os.path.join(output_checkpoint_path, f"layer_{final_layer}-model_00-model_states.pt"))
    del loaded_tps
    pbar.update(1)

    # Load output embedding
    pbar.set_description(f"Merging output embedding")

    # TODO: change variable 48 => num_layers + 4
    final_layer += 1
    loaded_tps = [
        torch.load(os.path.join(input_checkpoint_path, f"layer_{final_layer}-model_{i:02d}-model_states.pt")) 
        for i in range(mp_size)
    ]

    merged = {}
    for k in ["final_linear.weight"]:
        merged[k] = torch.cat([tp[k] for tp in loaded_tps], dim=0)
    # TODO: change variable 48 => num_layers + 4
    torch.save(merged, os.path.join(output_checkpoint_path, f"layer_{final_layer}-model_00-model_states.pt"))
    del loaded_tps
    pbar.update(1)
    pbar.set_description("Done.")


def merge(config_path, vocab_size, input_dir, output_dir):
    latest_info_file = os.path.join(input_dir, "latest")
    with open(latest_info_file, 'r') as f:
        latest_step = f.readline()

    input_checkpoint_path = os.path.join(input_dir, latest_step)
    output_checkpoint_path = os.path.join(output_dir, latest_step)
    os.makedirs(output_checkpoint_path, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "configs"), exist_ok=True)

    with open(config_path, 'r') as f:
        parallel_config = yaml.load(f, Loader=yaml.FullLoader)
    
    mp_size = parallel_config.get('model-parallel-size', 1)
    for i in range(mp_size):
        modify_model_states(
            input_model_state_path=os.path.join(
                input_checkpoint_path, f"mp_rank_{i:02d}_model_states.pt"),
            output_model_state_path=os.path.join(
                output_checkpoint_path, f"mp_rank_{i:02d}_model_states.pt"),
            vocab_size=vocab_size
        )
    modify_config(
        input_config_path=config_path,
        output_config_path=os.path.join(output_dir, "configs", "cofnig.yml"),
        output_dir=output_dir,
        vocab_size=vocab_size
    )
    merge_model_weights(
        config=parallel_config,
        input_checkpoint_path=input_checkpoint_path,
        output_checkpoint_path=output_checkpoint_path,
    )
    shutil.copyfile(
        parallel_config.get('vocab-file', './tokenizer.json'),
        os.path.join(output_dir, "tokenizer.json"),
    )
    with open(os.path.join(output_dir, "latest"), "w") as f:
        f.write(latest_step)


def main():
    parser = argparse.ArgumentParser(description='Merge 20B checkpoint.')
    parser.add_argument('--config_file', type=str,
                        help='yml config file path (not relative path), which should be ~.yml')
    parser.add_argument('--vocab_size', type=int, help='set vocab size of models.')
    parser.add_argument('--input_dir', type=str,
                    help='Checkpoint dir, which should contain (e.g. a folder named "global_step150000")')
    parser.add_argument('--output_dir', type=str,
                        help='Output dir, to save the 1-GPU weights configs')
    args = parser.parse_args()
    merge(args.config_file, args.vocab_size, args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
