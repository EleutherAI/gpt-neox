"""
script to merge or split model parallel checkpoints 

This script...
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
    args = parser.parse_args()

   
    checkpoint_dir = Path(args.checkpoint_dir)
    assert checkpoint_dir.is_dir(), "checkpoint dir does not exist: "+str(checkpoint_dir)

    if args.global_step is None:
        if os.path.isfile(checkpoint_dir / "latest"):
            with open(checkpoint_dir / "latest") as f:
                args.global_step = int(f.read().strip().replace("global_step", ""))
        else:
            raise ValueError("No global step provided")

    args.weights_dir = checkpoint_dir / f"global_step{args.global_step}"

    args.output_dir = Path(args.output_dir)
    args.output_weights_dir = args.output_dir / f"global_step{args.global_step}"
    args.output_configs_dir = args.output_weights_dir / "configs"

    return args

def get_output_config(args: argparse.Namespace):
    """
    read config files from source directory and change values to match the target for inference
    """

    # load all config files
    result = dict()
    for config in (args.weights_dir / "configs").glob("*.yml"):
        with open(config) as f:
            data = yaml.full_load(f)
            result.update(data)

    # update model parallel size dependent on args
    if "model-parallel-size" in result:
        result["model-parallel-size"] = args.model_parallel
    elif "model_parallel_size" in result:
        result["model_parallel_size"] = args.model_parallel

    # update pipe parallel size dependent on args
    if "pipe-parallel-size" in result:
        result["pipe-parallel-size"] = args.pipe_parallel
    elif "pipe_parallel_size" in result:
        result["pipe_parallel_size"] = args.pipe_parallel

    # remove zero optimizer
    # Loading a zero optimizer in inference results in an error due to attempted weight load
    if "zero_optimization" in result:
        if "stage" in result["zero_optimization"]:
            result["zero_optimization"]["stage"] = 0
    if "zero-optimization" in result:
        if "stage" in result["zero-optimization"]:
            result["zero-optimization"]["stage"] = 0

    return result

def get_weight_paths_by_layer(args: argparse.Namespace):

    # load list of source weight files
    paths = list(args.weights_dir.glob("*.pt"))
    
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
    Check if all tensors in a list is equal
    """
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(torch.allclose(first, x) for x in iterator)

if __name__ == "__main__":
    # get arguments
    args = parse_args()
    print(f"* Merging from {args.weights_dir}", flush=True)
    print(f"* Merging to {args.output_weights_dir}", flush=True)

    # load modified configs
    config = get_output_config(args)

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
    if args.output_weights_dir.is_dir():
        print("* Output weights dir already exists. It is deleted to guarantee integrity.", flush=True)
        shutil.rmtree(args.output_weights_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_weights_dir, exist_ok=True)
    os.makedirs(args.output_configs_dir, exist_ok=True)

    # save modified config
    with open(args.output_configs_dir / "config.yml", "w") as f:
        json.dump(config, f, indent=4)

    # load weight paths grouped by layer
    # so that we can merge layer by layer
    weight_paths_by_layer = get_weight_paths_by_layer(args)


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
        for mp in range(args.model_parallel):
            for layer_name, partitions in grouped_weights.items():
                if layer == "model_states":
                    if layer_name in IGNORED_LAYERS:
                        # don't copy over optimizer / rng states as they won't be valid
                        continue
                    elif layer_name == "mp_world_size":
                        # overwrite  mp
                        out_sd[layer_name] = args.model_parallel
                    elif layer_name == "args":
                        # change mp size in sd args
                        p = partitions[0]
                        if "model-parallel-size" in p:
                            p["model-parallel-size"] = args.model_parallel
                        elif "model_parallel_size" in p:
                            p["model_parallel_size"] = args.model_parallel
                    else:
                        out_sd[layer_name] = partitions[0]
                elif all_equal(partitions):
                    out_sd[layer_name] = partitions[0]
                else:
                    partition_dim = PARTITION_DIM_MAP.get(layer_name, 0)
                    merged = merge_partitions(partitions, partition_dim=partition_dim, mp=args.model_parallel, current_mp=mp)
                    out_sd[layer_name] = merged
            
            # save output
            if layer == "model_states":
                for pp in range(args.pipe_parallel):
                    out_path = (
                        args.output_weights_dir
                        / f"mp_rank_{str(pp).zfill(2)}_model_states.pt"
                    )
                    torch.save(out_sd, out_path)
                    print(f"* merged weights for layer {layer} and pp {pp}")

            else:
                out_path = (
                    args.output_weights_dir
                    / f"layer_{layer}-model_{str(mp).zfill(2)}-model_states.pt"
                )
                torch.save(out_sd, out_path)
                print(f"* merged weights for layer {layer} and mp {mp}")

    print("* DONE!")
