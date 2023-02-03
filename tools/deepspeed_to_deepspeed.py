#!/usr/bin/env python
import argparse
import os
from pathlib import Path
import sys
import torch
import yaml


# insert megatron's root dir into sys.path
root_repo_path = str(Path(__file__).resolve().parents[1])
if root_repo_path not in sys.path:
    sys.path.insert(0, root_repo_path)

from megatron.tokenizer.tokenizer import _vocab_size_with_padding
from megatron.neox_arguments import NeoXArgs
from deepspeed.checkpoint.deepspeed_checkpoint import (
    ARGS_KEY,
    CHECKPOINT_INFO_KEY,
)

from deepspeed.checkpoint import (
    DeepSpeedCheckpoint,
    NeoxCheckpoint,
    get_model_ckpt_name_for_rank,
    get_zero_ckpt_name_for_rank,
    get_layer_ckpt_name_for_rank,
)

CHECKPOINT_FILE_SUFFIX = "_model_states.pt"
MP_WORLD_SIZE = "mp_world_size"
WORD_EMBEDDINGS_KEY = "word_embeddings.weight"
ORIGINAL_VOCAB_SIZE = "original_vocab_size"
PADDED_VOCAB_SIZE = "padded_vocab_size"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder",
        default=None,
        type=str,
        help="Input DeepSpeed Checkpoint folder",
    )
    parser.add_argument(
        "--output_folder",
        default=None,
        type=str,
        help="Output Megatron checkpoint folder",
    )
    parser.add_argument("--config", type=str, help="Path to yml config")
    parser.add_argument("--target_tp", default=None, type=int, help="Target TP degree")
    parser.add_argument("--target_pp", default=None, type=int, help="Target PP degree")
    parser.add_argument("--target_dp", default=None, type=int, help="Target DP degree")
    args = parser.parse_args()
    print(f"args = {args}")
    return args


def _save_checkpoint(file_path, chkpt_sd):
    dir, _ = os.path.split(file_path)
    os.makedirs(dir, exist_ok=True)
    torch.save(chkpt_sd, file_path)


def _create_transformer_layer_checkpoint(
    ds_checkpoint, base_folder, tp_index, pp_index
):
    sd_list = ds_checkpoint.get_transformer_state(tp_index, pp_index)
    layer_id_list = ds_checkpoint.get_pp_transformer_map(pp_index)
    assert len(sd_list) == len(layer_id_list)
    for sd, layer_id in zip(sd_list, layer_id_list):
        ckpt_path = get_layer_ckpt_name_for_rank(
            base_folder=base_folder, layer_id=layer_id, tp_rank=tp_index
        )
        _save_checkpoint(ckpt_path, sd)


def _strip_vocab_padding(ds_checkpoint, padded_vocab_tensor, tokenizer):
    target_args = ds_checkpoint.get_args()
    # checkpoint_info = ds_checkpoint.get_checkpoint_info()

    target_args["tensor_model_parallel_size"] = ds_checkpoint.tp_degree
    target_args[PADDED_VOCAB_SIZE] = _vocab_size_with_padding(
        tokenizer.vocab_size, target_args
    )
    assert target_args[PADDED_VOCAB_SIZE] <= padded_vocab_tensor.numel()
    # checkpoint_info[PADDED_VOCAB_SIZE] = target_args[PADDED_VOCAB_SIZE]
    unpadded_vocab_tensor = torch.narrow(
        padded_vocab_tensor, 0, 0, target_args[PADDED_VOCAB_SIZE]
    )
    return unpadded_vocab_tensor.clone()


def _create_embedding_layer_checkpoint(ds_checkpoint, base_folder, tp_index, tokenizer):
    sd = ds_checkpoint.get_embedding_state(tp_index)
    if ds_checkpoint.is_change_tp_degree():
        sd[WORD_EMBEDDINGS_KEY] = _strip_vocab_padding(
            ds_checkpoint, sd[WORD_EMBEDDINGS_KEY], tokenizer
        )
    layer_id = ds_checkpoint.get_embedding_layer_id()
    ckpt_path = get_layer_ckpt_name_for_rank(
        base_folder=base_folder, tp_rank=tp_index, layer_id=layer_id
    )
    _save_checkpoint(ckpt_path, sd)


def _create_final_norm_layer_checkpoint(ds_checkpoint, base_folder, tp_index):
    sd = ds_checkpoint.get_final_norm_state(tp_index)
    layer_id = ds_checkpoint.get_final_norm_layer_id()
    ckpt_path = get_layer_ckpt_name_for_rank(
        base_folder=base_folder, tp_rank=tp_index, layer_id=layer_id
    )
    _save_checkpoint(ckpt_path, sd)


def _create_2d_parallel_checkpoint(ds_checkpoint, base_folder, tp_index, pp_index):
    sd = ds_checkpoint.get_2d_parallel_state(tp_index=tp_index, pp_index=pp_index)
    sd[MP_WORLD_SIZE] = ds_checkpoint.tp_degree
    file_id = pp_index * ds_checkpoint.tp_degree + tp_index
    ckpt_path = get_model_ckpt_name_for_rank(base_folder, f"{file_id:02d}")

    # Adjust specific fields
    sd[ARGS_KEY] = ds_checkpoint.get_args()
    sd[ARGS_KEY]["tensor_model_parallel_size"] = ds_checkpoint.tp_degree
    sd[ARGS_KEY]["pipeline_model_parallel_size"] = ds_checkpoint.pp_degree
    sd[CHECKPOINT_INFO_KEY][PADDED_VOCAB_SIZE] = sd[ARGS_KEY].padded_vocab_size
    _save_checkpoint(ckpt_path, sd)


def _create_zero_checkpoint(ds_checkpoint, base_folder, dp_index, pp_index, tp_index):
    _2d_rank = (pp_index * ds_checkpoint.tp_degree) + tp_index
    sd = ds_checkpoint.get_zero_checkpoint_state(
        pp_index=pp_index, tp_index=tp_index, dp_index=dp_index
    )

    ckpt_path = get_zero_ckpt_name_for_rank(
        base_folder=base_folder, dp_rank=dp_index, mp_rank=_2d_rank
    )
    _save_checkpoint(ckpt_path, sd)


def _create_latest_file(base_folder, file_name, latest_tag):
    file_path = os.path.join(base_folder, file_name)
    os.makedirs(base_folder, exist_ok=True)
    with open(file_path, "w") as f:
        f.write(str(latest_tag))


def main():
    print(f"Convert DeepSpeed Checkpoint to DeepSpeed Checkpoint")

    args = parse_arguments()
    print(
        f"Converting DeepSpeed checkpoint in {args.input_folder} to DeepSpeed checkpoint in {args.output_folder}"
    )

    neox_args = NeoXArgs.from_ymls([args.config])
    tokenizer = neox_args.build_tokenizer()

    ds_checkpoint = NeoxCheckpoint(
        args.input_folder, args.target_tp, args.target_pp, args.target_dp
    )
    iteration = ds_checkpoint.get_iteration()
    latest_tag = f"global_step{iteration}"
    _create_latest_file(
        args.output_folder, "latest_checkpointed_iteration.txt", iteration
    )
    _create_latest_file(args.output_folder, "latest", latest_tag)
    base_folder = os.path.join(args.output_folder, latest_tag)

    for i in range(ds_checkpoint.tp_degree):
        _create_embedding_layer_checkpoint(ds_checkpoint, base_folder, i, tokenizer)
        _create_final_norm_layer_checkpoint(ds_checkpoint, base_folder, i)

        for j in range(ds_checkpoint.pp_degree):
            _create_transformer_layer_checkpoint(ds_checkpoint, base_folder, i, j)
            _create_2d_parallel_checkpoint(ds_checkpoint, base_folder, i, j)

    for i in range(ds_checkpoint.dp_degree):
        for j in range(ds_checkpoint.pp_degree):
            for k in range(ds_checkpoint.tp_degree):
                _create_zero_checkpoint(ds_checkpoint, base_folder, i, j, k)


if __name__ == "__main__":
    main()
