#!/usr/bin/env python
import argparse
import os
from pathlib import Path
import sys
import torch
import yaml


# insert megatron's root dir into sys.path
root_repo_path = str(Path(__file__).resolve().parents[2])
if root_repo_path not in sys.path:
    sys.path.insert(0, root_repo_path)

from megatron.neox_arguments import NeoXArgs
from deepspeed.checkpoint.deepspeed_checkpoint import (
    ARGS_KEY,
    CHECKPOINT_INFO_KEY,
)

from deepspeed.checkpoint import (
    NeoxCheckpoint,
    get_model_ckpt_name_for_rank,
    get_zero_ckpt_name_for_rank,
    get_layer_ckpt_name_for_rank,
)

CHECKPOINT_FILE_SUFFIX = "_model_states.pt"
MP_WORLD_SIZE = "mp_world_size"
WORD_EMBEDDINGS_KEY = "word_embeddings.weight"
FINAL_LINEAR_KEY = "final_linear.weight"
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
    parser.add_argument(
        "--iteration",
        default=None,
        type=int,
        help="Which checkpoint to load, defaults to what is in latest if None",
    )

    args = parser.parse_args()
    print(f"args = {args}")
    return args


def _vocab_size_with_padding(orig_vocab_size, divisible_by, tp_size):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = divisible_by * tp_size
    while (after % multiple) != 0:
        after += 1

    print(
        " > padded vocab (size: {}) with {} dummy tokens "
        "(new size: {})".format(orig_vocab_size, after - orig_vocab_size, after),
        flush=True,
    )
    return after


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


def _strip_vocab_padding(ds_checkpoint, padded_vocab_tensor, neox_args):
    target_args = ds_checkpoint.get_args()
    target_args["model_parallel_size"] = ds_checkpoint.tp_degree

    padded_vocab_size = _vocab_size_with_padding(
        neox_args.tokenizer.vocab_size,
        target_args["make_vocab_size_divisible_by"],
        ds_checkpoint.tp_degree,
    )
    padded_layer_size = padded_vocab_size // ds_checkpoint.tp_degree
    assert padded_vocab_size <= padded_vocab_tensor.numel()
    target_args[PADDED_VOCAB_SIZE] = padded_vocab_size
    unpadded_vocab_tensor = torch.narrow(padded_vocab_tensor, 0, 0, padded_layer_size)
    return unpadded_vocab_tensor.clone()


def _create_embedding_layer_checkpoint(ds_checkpoint, base_folder, tp_index, args):
    sd = ds_checkpoint.get_embedding_state(tp_index)
    if ds_checkpoint.is_change_tp_degree():
        print(f"TP index: {tp_index}, embeddings shape {sd[WORD_EMBEDDINGS_KEY].shape}")
        sd[WORD_EMBEDDINGS_KEY] = _strip_vocab_padding(
            ds_checkpoint, sd[WORD_EMBEDDINGS_KEY], args
        )
    layer_id = ds_checkpoint.get_embedding_layer_id()
    ckpt_path = get_layer_ckpt_name_for_rank(
        base_folder=base_folder, tp_rank=tp_index, layer_id=layer_id
    )
    _save_checkpoint(ckpt_path, sd)


def _create_final_norm_layer_checkpoint(ds_checkpoint, base_folder, tp_index, args):
    sd = ds_checkpoint.get_final_norm_state(tp_index)
    layer_id = ds_checkpoint.get_final_norm_layer_id()
    if ds_checkpoint.is_change_tp_degree():
        sd[FINAL_LINEAR_KEY] = _strip_vocab_padding(
            ds_checkpoint, sd[FINAL_LINEAR_KEY], args
        )
    ckpt_path = get_layer_ckpt_name_for_rank(
        base_folder=base_folder, tp_rank=tp_index, layer_id=layer_id
    )
    _save_checkpoint(ckpt_path, sd)


def _create_2d_parallel_checkpoint(ds_checkpoint, base_folder, tp_index, pp_index):
    sd = ds_checkpoint.get_2d_parallel_state(tp_index=tp_index, pp_index=pp_index)
    ckpt_info = ds_checkpoint.get_checkpoint_info()
    sd[MP_WORLD_SIZE] = ds_checkpoint.tp_degree
    file_id = pp_index * ds_checkpoint.tp_degree + tp_index
    ckpt_path = get_model_ckpt_name_for_rank(base_folder, f"{file_id:02d}")

    # Adjust specific fields
    sd[ARGS_KEY] = ds_checkpoint.get_args()
    # sd[ARGS_KEY][PADDED_VOCAB_SIZE] = ckpt_info[PADDED_VOCAB_SIZE]
    sd[ARGS_KEY]["model_parallel_size"] = ds_checkpoint.tp_degree
    sd[ARGS_KEY]["pipe_parallel_size"] = ds_checkpoint.pp_degree
    if CHECKPOINT_INFO_KEY not in sd:
        sd[CHECKPOINT_INFO_KEY] = {}
    sd[CHECKPOINT_INFO_KEY][PADDED_VOCAB_SIZE] = sd[ARGS_KEY][PADDED_VOCAB_SIZE]
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


def get_folder(args):
    folder = Path(args.input_folder)
    if args.iteration is None:
        with open(folder / "latest") as latest_file:
            tag = latest_file.read().strip()
    else:
        tag = f"global_step{args.iteration}"
    return folder / tag


def main():
    print(f"Convert DeepSpeed Checkpoint to DeepSpeed Checkpoint")

    args = parse_arguments()
    print(
        f"Converting DeepSpeed checkpoint in {args.input_folder} to DeepSpeed checkpoint in {args.output_folder}"
    )

    neox_args = NeoXArgs.from_ymls([args.config])
    neox_args.build_tokenizer()

    ckpt_folder = get_folder(args)

    ds_checkpoint = NeoxCheckpoint(
        ckpt_folder, args.target_tp, args.target_pp, args.target_dp
    )
    iteration = ds_checkpoint.get_iteration()
    latest_tag = f"global_step{iteration}"
    _create_latest_file(
        args.output_folder, "latest_checkpointed_iteration.txt", iteration
    )
    _create_latest_file(args.output_folder, "latest", latest_tag)
    base_folder = os.path.join(args.output_folder, latest_tag)

    for i in range(ds_checkpoint.tp_degree):
        _create_embedding_layer_checkpoint(ds_checkpoint, base_folder, i, neox_args)
        _create_final_norm_layer_checkpoint(ds_checkpoint, base_folder, i, neox_args)

        for j in range(ds_checkpoint.pp_degree):
            _create_transformer_layer_checkpoint(ds_checkpoint, base_folder, i, j)
            _create_2d_parallel_checkpoint(ds_checkpoint, base_folder, i, j)

    for i in range(ds_checkpoint.dp_degree):
        for j in range(ds_checkpoint.pp_degree):
            for k in range(ds_checkpoint.tp_degree):
                _create_zero_checkpoint(ds_checkpoint, base_folder, i, j, k)


if __name__ == "__main__":
    main()
