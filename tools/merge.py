"""
script to merge or split model parallel checkpoints 

This script:
- assumes a checkpoint directory with pipeline parallel checkpoints (i.e. a global_step directory with files named like 'layer_00-model_00-model_states.pt')
- assumes that the checkpoint names haven't been changed, since it makes the script much cleaner. If you've changed the name from the default `layer_nn-model_nn-model_states.pt` pattern - *this script will not work*
- assumes the config files are saved to a subdirectory in the global_step directory
- merges the model parallel files of a single layer (i.e. joins 'layer_00-model_00-model_states.pt' and 'layer_00-model_01-model_states.pt')
- potentially splits the merged checkpoint to a target model parallel size
- does not change pipeline parallel settings; You might want to adjust when reloading a checkpoint.

Examples
```console

# print help
python tools/merge.py --help

# merge the global_step10 checkpoint in the checkpoints directory to checkpoints_merged with output model parallel 1 and pipe parallel 2
python tools/merge.py -d checkpoints -o checkpoints_merged -s 10 -mp 1 -pp 2

# merge the global_step10 checkpoint in the checkpoints directory to checkpoints_merged with output model parallel 4 and pipe parallel 1
python tools/merge.py -d checkpoints -o checkpoints_merged -s 10 -mp 4 -pp 1
```
"""
import re
import os
import yaml
import json
import shutil
import argparse
from typing import List
from pathlib import Path
from collections import defaultdict

import torch

# weights of these layers will not be copied as they won't be valid
IGNORED_LAYERS = [
                    "optimizer",
                    "random_rng_state",
                    "np_rng_state",
                    "torch_rng_state",
                    "cuda_rng_state",
                    "rng_tracker_states",
                    "dp_world_size"
                ]

def parse_args():
    parser = argparse.ArgumentParser(
        "Merge or split model parallel groups in a pretrained model"
    )
    parser.add_argument(
        "-d",
        "--checkpoint_dir",
        default="checkpoints/",
        type=str,
        help="parent directory in which checkpoints are stored; this is the 'save' parameter of neox args",
    )
    parser.add_argument(
        "-s",
        "--global_step",
        type=int,
        default=None,
        help='which global step to edit (each checkpoint dir should contain multiple global steps.) \
                                                                        defaults to the global step contained in "checkpoint_dir/latest"',
    )
    parser.add_argument(
        "-mp",
        "--model_parallel",
        type=int,
        default=1,
        help='number of model parallel partitions in the output',
    )
    parser.add_argument(
        "-pp",
        "--pipe_parallel",
        type=int,
        default=1,
        help='number of pipe parallel partitions in the output',
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="checkpoints_merged",
        type=str,
        help="Where to save the merged model",
    )

    return parser.parse_args()


def replace(d, key, value): 
    # function to replace a k/v pair in a dict, that's agnostic to the difference between '-' and '_'
    k_alt = key.replace('-', '_')
    if key in d:
        d[key] = value
    elif k_alt in d:
        d[k_alt] = value

def get_output_config(checkpoint_dir, model_parallel_size, pipe_parallel_size, remove_zero_optimizer=True):
    """
    read config files from source directory and change values to match the desired output.

    Args:
        checkpoint_dir: the original checkpoint directory
        model_parallel_size: the output model parallel size
        pipe_parallel_size: the output pipe parallel_size
        remove_zero_optimizer: remove zero optimizer settings from the output dict. 
                               Not doing so may result in errors due to size mismatches in optimizer states.
    """

    # load all config files
    result = dict()
    for config in (checkpoint_dir / "configs").glob("*.yml"):
        with open(config) as f:
            data = yaml.full_load(f)
            result.update(data)

    # update model parallel size dependent on args
    replace(result, "model-parallel-size", model_parallel_size)
    replace(result, "pipe-parallel-size", pipe_parallel_size)
    
    # remove zero optimizer
    # Loading a zero optimizer in inference results in an error due to attempted weight load
    if remove_zero_optimizer:
        if "zero_optimization" in result:
            if "stage" in result["zero_optimization"]:
                result["zero_optimization"]["stage"] = 0
        if "zero-optimization" in result:
            if "stage" in result["zero-optimization"]:
                result["zero-optimization"]["stage"] = 0

    return result

def get_weight_paths_by_layer(weights_dir: Path) -> List[Path]:

    # load list of source weight files
    paths = list(weights_dir.glob("*.pt"))
    
    # group checkpoint paths by layer index
    paths_by_layer = defaultdict(list)
    for p in paths:
        layer = re.search("layer_(\d+)", p.stem)
        if layer is None:
            layer = "model_states"
        else:
            layer = layer.group(1)
        paths_by_layer[layer].append(p) 

    # sort by layer name to have print statements below ordered
    result = [(l, ps) for l, ps in paths_by_layer.items()]
    result = sorted(result, key=lambda i: i[0])

    return result

def load_grouped_weights(weight_paths: List[Path]):
    """
    Loads a dictinary mapping layer name to a list of weights 
    """
    
    # Important! Sort by name (i.e. the model parallel index)
    # This guarantees the right order of weights in the merged matrix
    weight_paths = sorted(weight_paths, key=lambda i: i.name)

    # load checkpoints to cpu + group by layer name
    loaded = []
    by_layer_name = defaultdict(list)
    for weight_path in weight_paths:
        loaded.append(torch.load(weight_path, map_location="cpu"))
        
    for l in loaded:
        for k, v in l.items():
            by_layer_name[k].append(v)

    return by_layer_name

@torch.no_grad()
def merge_partitions(partitions, partition_dim, stride=1, mp=1, current_mp=None):
    # Number and size of each partition.
    num_partitions = len(partitions)
    per_partition_size = None
    for partition in partitions:
        if per_partition_size is None:
            per_partition_size = partition.shape[partition_dim]
        else:
            assert per_partition_size == partition.size(
                partition_dim
            ), "all partitions should be of equal size"

    merged_size = list(partition.shape)
    merged_size[partition_dim] *= len(partitions)
    merged = torch.zeros(*merged_size).to(partitions[0].dtype)

    if stride == 1:
        assert (per_partition_size * num_partitions) == merged.size(
            partition_dim
        ), "ERROR: sizes do not match."
        # If stride is 1, then do simple concatination.
        torch.cat(partitions, dim=partition_dim, out=merged)

        if mp == 1:
            return merged
        else:
            assert merged.shape[partition_dim] % mp == 0, "cannot convert to mp size "+str(mp)
            mp_size = merged.shape[partition_dim] // mp 
            splits = torch.split(merged, mp_size, dim=partition_dim)
            assert len(splits) == mp, "got different number of splits than mp"
            for i in range(len(splits)):
                assert len(splits[i].shape) == len(merged.shape), "split has different dimensions than merged"
                assert list(splits[0].shape) == list(splits[i].shape), "all splits should have equal size"
            return splits[current_mp]
    else:
        # we don't use stride > 1 anywhere rn
        raise NotImplementedError

def all_equal(iterator):
    """
    Check if all tensors in an iterator are equal
    """
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(torch.allclose(first, x) for x in iterator)

def merge_checkpoints(checkpoint_dir, model_parallel_size, pipe_parallel_size, output_dir, global_step=None):
    checkpoint_dir = Path(checkpoint_dir)
    assert checkpoint_dir.is_dir(), f"checkpoint dir does not exist: {str(checkpoint_dir)}"
    output_dir = Path(output_dir)

    if global_step is None:
        if os.path.isfile(checkpoint_dir / "latest"):
            with open(checkpoint_dir / "latest") as f:
                global_step = int(f.read().strip().replace("global_step", ""))
        else:
            raise ValueError("No global step provided")

    weights_dir = checkpoint_dir / f"global_step{global_step}"

    
    output_weights_dir = output_dir / f"global_step{global_step}"
    output_configs_dir = output_dir / "configs"

    print(f"* Merging from {weights_dir}", flush=True)
    print(f"* Merging to {output_weights_dir}", flush=True)

    # load modified configs
    config = get_output_config(checkpoint_dir, model_parallel_size, pipe_parallel_size)

    # create partition dim map dependent on config
    # maps layer names to their partitioned dimension
    # if the layer name isn't in this list, we assume the partitioned dimension is 0

    output_layer_parallelism = config.get("output_layer_parallelism") or config.get("output-layer-parallelism") or "row"
    PARTITION_DIM_MAP = {
        "attention.dense.weight": 1,
        "mlp.dense_4h_to_h.weight": 1,
        "final_linear.weight": 0 if output_layer_parallelism == "column" else 1,
    }

    # prepare output directories
    if output_weights_dir.is_dir():
        resp = input(f"* Output weights dir ({output_weights_dir}) already exists. Do you want to overwrite it? (yes/no) ")
        if resp.lower() in ['yes', 'y']:
            shutil.rmtree(output_weights_dir)
        else:
            exit()

    for p in [output_weights_dir, output_configs_dir]:
        p.mkdir(exist_ok=True, parents=True)

    # save modified config
    with open(output_configs_dir / "config.yml", "w") as f:
        json.dump(config, f, indent=4)

    # load weight paths grouped by layer
    # so that we can merge layer by layer
    weight_paths_by_layer = get_weight_paths_by_layer(weights_dir)


    # iterate over layers and produce a merged checkpoint
    for (layer, weight_paths) in weight_paths_by_layer:

        # load weights grouped by module name for the current layer
        grouped_weights = load_grouped_weights(weight_paths)

        # merge and save
        # We merge and iterate for every output model parallel rank.
        # This implies duplicate merge operations but facilitates
        # the code due to multiple possible to be merged module weights within
        # each checkpoint file.
        # The one time merge can live with a little inefficiency.
        out_sd = {}
        for mp in range(model_parallel_size):
            for layer_name, partitions in grouped_weights.items():
                if layer == "model_states":
                    if layer_name in IGNORED_LAYERS:
                        # don't copy over optimizer / rng states as they won't be valid
                        continue
                    elif layer_name == "mp_world_size":
                        # overwrite mp in sd
                        out_sd[layer_name] = model_parallel_size
                    elif layer_name == "args":
                        # change mp size in sd args
                        p = partitions[0]
                        replace(p, "model-parallel-size", model_parallel_size)
                    else:
                        out_sd[layer_name] = partitions[0]
                elif all_equal(partitions): # if all partitions are equal anyway, we can naively merge them
                    out_sd[layer_name] = partitions[0]
                else:
                    partition_dim = PARTITION_DIM_MAP.get(layer_name, 0)
                    merged = merge_partitions(partitions, partition_dim=partition_dim, mp=model_parallel_size, current_mp=mp)
                    out_sd[layer_name] = merged
            
            # save output
            if layer == "model_states":
                for pp in range(args.pipe_parallel):
                    out_path = (
                        output_weights_dir
                        / f"mp_rank_{str(pp).zfill(2)}_model_states.pt"
                    )
                    torch.save(out_sd, out_path)
                    print(f"* merged weights for layer {layer} and pp {pp}")

            else:
                out_path = (
                    output_weights_dir
                    / f"layer_{layer}-model_{str(mp).zfill(2)}-model_states.pt"
                )
                torch.save(out_sd, out_path)
                print(f"* merged weights for layer {layer} and mp {mp}")

    print("* DONE!")

if __name__ == "__main__":
    # get arguments
    args = parse_args()
    merge_checkpoints(args.checkpoint_dir, args.model_parallel, args.pipe_parallel, args.output_dir, args.global_step)
    
