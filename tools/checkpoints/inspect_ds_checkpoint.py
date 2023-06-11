import sys
from pathlib import Path

# insert megatron's root dir into sys.path
root_repo_path = str(Path(__file__).resolve().parents[2])
if root_repo_path not in sys.path:
    sys.path.insert(0, root_repo_path)

import argparse

from deepspeed.checkpoint import NeoxCheckpoint


def list_files(file_list, tag):
    print(f"Listing files: {tag}")
    for i, file in enumerate(file_list):
        print(f"{i+1}: {file}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder", default=None, type=str, help="DeepSpeed Checkpoint folder"
    )
    parser.add_argument("--target_tp", default=None, type=int, help="Target TP degree")
    parser.add_argument("--target_pp", default=None, type=int, help="Target PP degree")
    parser.add_argument("--target_dp", default=None, type=int, help="Target DP degree")
    parser.add_argument(
        "--iteration", default=None, type=int, help="Which iteration to load"
    )
    args = parser.parse_args()
    print(f"args = {args}")
    return args


def show_input_files(ds_checkpoint):
    list_files(ds_checkpoint.file_list, "all")
    list_files(ds_checkpoint.zero_files, "zero")
    list_files(ds_checkpoint.layer_files, "layer")
    list_files(ds_checkpoint.mp_rank_files, "mp rank")


def show_simple_state(ds_checkpoint):
    print(f"layer keys = {ds_checkpoint.layer_keys}")
    print(f"layer count = {ds_checkpoint.layer_count}")

    print(
        f"tp_degree_count = {ds_checkpoint.original_tp_degree} ------> {ds_checkpoint.tp_degree}"
    )
    print(
        f"pp_degree_count = {ds_checkpoint.original_pp_degree} ------> {ds_checkpoint.pp_degree}"
    )
    print(
        f"dp_degree_count = {ds_checkpoint.original_dp_degree} ------> {ds_checkpoint.dp_degree}"
    )
    ds_checkpoint.old_2d_map.print_data("old 2d map ==>")
    ds_checkpoint.new_2d_map.print_data("new 2d map ==>")


def show_mappings(ds_checkpoint):
    ds_checkpoint.show_pp_transformer_map()
    ds_checkpoint.show_transformer_file_map()
    ds_checkpoint.show_tp_embedding_map()
    ds_checkpoint.show_tp_final_norm_map()
    ds_checkpoint.show_2d_mapping()


def show_state_summary(tag, sd):
    summary = {k: v.shape for k, v in sd.items()}
    print(f"{tag} = {summary}")


def show_embedding_states(ds_checkpoint):
    for i in range(0, ds_checkpoint.tp_degree):
        sd = ds_checkpoint.get_embedding_state(i)
        show_state_summary(f"embedding[{i}]", sd)


def show_final_norm_states(ds_checkpoint):
    for i in range(0, ds_checkpoint.tp_degree):
        sd = ds_checkpoint.get_final_norm_state(i)
        show_state_summary(f"final_norm[{i}]", sd)


def show_transformer_states(ds_checkpoint):
    for i in range(0, ds_checkpoint.tp_degree):
        for j in range(0, ds_checkpoint.pp_degree):
            state_list = ds_checkpoint.get_transformer_state(tp_index=i, pp_index=j)
            print(f"tp_pp_rank[{i},{j}] = ")
            for k, sd in enumerate(state_list):
                show_state_summary(f"      block[{k}]", sd)
                print("")


def get_folder(args):
    folder = Path(args.folder)
    if args.iteration is None:
        with open(folder / "latest") as latest_file:
            tag = latest_file.read()
    else:
        tag = f"global_step{args.iteration}"
    return folder / tag


def main():
    print(f"Inspecting DeepSpeed Checkpoint")
    args = parse_arguments()

    ckpt_folder = get_folder(args)

    ds_checkpoint = NeoxCheckpoint(
        ckpt_folder, args.target_tp, args.target_pp, args.target_dp
    )
    ds_checkpoint.validate_files()

    show_simple_state(ds_checkpoint)
    show_input_files(ds_checkpoint)
    show_simple_state(ds_checkpoint)
    show_mappings(ds_checkpoint)
    show_embedding_states(ds_checkpoint)
    show_final_norm_states(ds_checkpoint)
    show_transformer_states(ds_checkpoint)
    checkpoint_args = ds_checkpoint.get_args()
    print(f"checkpoint args = {checkpoint_args}")


if __name__ == "__main__":
    main()
