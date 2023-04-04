'''Copyright The Microsoft DeepSpeed Team'''

import os
import torch
import time
import deepspeed
import argparse
import pandas as pd
import numpy as np
from transformers import pipeline
from deepspeed.accelerator import get_accelerator

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

        df.to_csv(
            os.path.join(output_dir,
            "{}_{}_max_tokens_{}_world_size_{}_benchmark.csv".format(model.split('/')[-1], str(dtype).split('.')[1], max_tokens, world_size)))


def main(models, output_dir, use_deepspeed, dtype, graphs, kernel_inject, max_tokens, local_rank, world_size, trials):
    for model in models:
        benchmark_model(
            model, output_dir, use_deepspeed, dtype, graphs, kernel_inject, max_tokens, local_rank, world_size, trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='/home/mchorse/benchmarking/output', help="output_directory")
    parser.add_argument("--models", "-m", type=str, nargs='+', help="hf model names")
    parser.add_argument("--use_deepspeed", action="store_true", help="use deepspeed inference")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32", "int8"], help="int8, fp16, or fp32")
    parser.add_argument("--graphs", action="store_true", help="CUDA Graphs on")
    parser.add_argument("--kernel-inject", action="store_true", help="inject kernels on")
    parser.add_argument("--max-tokens", type=int, default=50, help="max new tokens")
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", "0")), help="local rank")
    parser.add_argument("--world_size", type=int, default=4, help="world size")
    parser.add_argument("--trials", type=int, default=30, help="number of trials")
    args = parser.parse_args()
    deepspeed.init_distributed()
    main(**vars(args))

