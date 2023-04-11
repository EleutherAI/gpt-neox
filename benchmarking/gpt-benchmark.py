'''Adapted from https://github.com/microsoft/DeepSpeed/blob/master/benchmarks/inference/gpt-bench.py'''

import argparse
import os
import time

import deepspeed
from deepspeed.accelerator import get_accelerator
import numpy as np
import pandas as pd
from transformers import pipeline
import torch
import yaml


def benchmark_model(
    model, output_dir, use_deepspeed, dtype, graphs, kernel_inject, max_tokens, local_rank, world_size, trials):

    deepspeed.init_distributed()
    if local_rank == 0:
        print("BENCHMARK SETTINGS:")
        print(f"\tMODEL: {model}")
        print(f"\tMAX_TOKENS: {max_tokens}")
        print(f"\tDTYPE: {dtype}")
        print(f"\tCUDA_GRAPHS: {graphs}")
        print(f"\tKERNEL_INJECT: {kernel_inject}")
        print(f"\tWORLD_SIZE: {world_size}")

    if dtype == "int8":
        dtype = torch.int8
    elif dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    pipe = pipeline("text-generation", model=model, framework="pt", device=local_rank)

    if dtype == torch.float16:
        pipe.model.half()
    print("")
    if use_deepspeed:
        pipe.model = deepspeed.init_inference(
            pipe.model,
            dtype=dtype,
            mp_size=world_size,
            replace_with_kernel_inject=kernel_inject,
            enable_cuda_graph=graphs,
        )
        pipe.model.profile_model_time()

    responses = []
    times = []
    mtimes = []
    for i in range(trials):
        get_accelerator().synchronize()
        start = time.time()
        r = pipe("DeepSpeed is", do_sample=False, max_new_tokens=max_tokens)
        get_accelerator().synchronize()
        end = time.time()
        responses.append(r)
        times.append(end - start)  # / (max_tokens - 3))
        if use_deepspeed:
            mtimes.append(sum(pipe.model.model_times()))

    if use_deepspeed:
        for_dataframe = np.vstack((times, mtimes, list(map(lambda t: t / (max_tokens - 3), times)))).T
        columns = ["(e2e) latency", "(model-only) latency", "(e2e) per token latency"]

    else:
        for_dataframe = np.vstack((times, list(map(lambda t: t / (max_tokens - 3), times)))).T
        columns = ["(e2e) latency", "(e2e) per token latency"]

    if local_rank == 0:
        df = pd.DataFrame(
            for_dataframe,
            columns = columns)

    deepspeed_str = "deepspeed" if use_deepspeed else "hf"
    deepspeed_dir = os.path.join(output_dir, deepspeed_str)
    max_tokens_dir = os.path.join(deepspeed_dir, "max_tokens_{}".format(max_tokens))
    world_size_dir = os.path.join(max_tokens_dir, "world_size_{}".format(world_size))

    os.makedirs(world_size_dir, exist_ok=True)

    fname = os.path.join(world_size_dir,
                           "{}_{}_benchmark.csv".format(model.split('/')[-1], str(dtype).split('.')[1]))
    
    print("saving benchmark to {}".format(fname))

    # save dataframe to CSV inside the directory for world_size
    df.to_csv(fname, index=False)


def main(models, output_dir, use_deepspeed, dtype, graphs, kernel_inject, max_tokens, local_rank, world_size, trials):
    for model in models:
        benchmark_model(
            model, output_dir, use_deepspeed, dtype, graphs, kernel_inject, max_tokens, local_rank, world_size, trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='/home/mchorse/benchmarking/output', help="output_directory")
    parser.add_argument("--config", type=str, default='configs/inference_test.yml')
    parser.add_argument("--use_deepspeed", action="store_true", help="use deepspeed inference")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32", "int8"], help="int8, fp16, or fp32")
    parser.add_argument("--graphs", action="store_true", help="CUDA Graphs on")
    parser.add_argument("--kernel-inject", action="store_true", help="inject kernels on")
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", "0")), help="local rank")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    models = config["models"]
    world_size = config["world_size"]
    trials = config["trials"]
    max_tokens = config["max_tokens"]

    main(models=models,
         output_dir=args.output_dir,
         use_deepspeed=args.use_deepspeed,
         dtype=args.dtype,
         graphs=args.graphs,
         kernel_inject=args.kernel_inject,
         max_tokens=max_tokens,
         local_rank=args.local_rank,
         world_size=world_size,
         trials=trials)

