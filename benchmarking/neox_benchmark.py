'''Adapted from https://github.com/microsoft/DeepSpeed/blob/master/benchmarks/inference/gpt-bench.py'''

import argparse
import os
import time

import deepspeed
from deepspeed.accelerator import get_accelerator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import pipeline
import torch
import yaml

# from megatron.initialize import initialize_megatron
# from megatron.neox_arguments import NeoXArgs
# from megatron.utils import print_rank_0, setup_for_inference_or_eval
# from megatron.text_generation_utils import generate_samples_from_prompt
# from megatron.training import setup_model_and_optimizer


PYTHIA_TO_OLD_SUFFIXES = {
    "70M": "19M",
    "160M": "125M",
    "410M": "350M",
    "1B": "800M",
    "1.4B": "1-3B",
    "2.8B": "2.7B",
    "6.9B": "6-7B",
    "12B": "13B",
    "20B": "20B"}


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

    pipe = pipeline("text-generation", model=model, framework="pt", device_map='auto')

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
    return df


def benchmark_gpt_neox(
        model, max_tokens, trials):
    
    model_suffix = model.split('/')[-1].split('-')[1]
    old_model_suffix = PYTHIA_TO_OLD_SUFFIXES[model_suffix]
    neox_args = NeoXArgs.from_ymls(
        ["/home/mchorse/gpt-neox/configs/{}.yml".format(old_model_suffix),
        "/home/mchorse/gpt-neox/configs/benchmark_setup.yml",
        "/home/mchorse/gpt-neox/configs/benchmarking.yml"])
    
    initialize_megatron(neox_args)

    model, _, _ = setup_model_and_optimizer(
        neox_args=neox_args,
        use_cache=True,
        iteration=neox_args.iteration,
    )  # we use setup_model_and_optimizer instead of get_model in order to initialize deepspeed

    print_rank_0("Finished loading model")

    prompts = ["DeepSpeed is" for x in range(trials)]

    generated_texts = generate_samples_from_prompt(
        neox_args=neox_args,
        model=model,
        text=prompts,
        eos_token_id=neox_args.eos_token_id,
        maximum_tokens=max_tokens,
        recompute=neox_args.recompute,
        temperature=neox_args.temperature,
        top_k=neox_args.top_k,
        top_p=neox_args.top_p,
    )
    print(generated_texts)
    

def main(models, output_dir, dtype, graphs, kernel_inject, max_tokens, local_rank, world_size, trials):
    deepspeed_dfs = []
    hf_dfs = []
    print("Models to benchmark: {}".format(models))
    for model in models:
        print("Benchmarking model: {}".format(model))
        # native gpt-neox inference
        print("Running with gpt-neox")
        #benchmark_gpt_neox(model, max_tokens, trials)
        # run using deepspeed
        print("Running with deepspeed")
        deepspeed_dfs.append(benchmark_model(
            model, output_dir, True, dtype, graphs, kernel_inject, max_tokens, local_rank, world_size, trials))

        # run using huggingface
        print("Running with huggingface")
        hf_dfs.append(benchmark_model(
            model, output_dir, False, dtype, graphs, kernel_inject, max_tokens, local_rank, world_size, trials))
        


    print("plotting results")
    # drop first 3 rows (warmup)
    ds_means = [x["(e2e) latency"].iloc[3:].mean() for x in deepspeed_dfs]
    ds_std = [x["(e2e) latency"].iloc[3:].std() for x in deepspeed_dfs]
    hf_means = [x["(e2e) latency"].iloc[3:].mean() for x in hf_dfs]
    hf_std = [x["(e2e) latency"].iloc[3:].std() for x in hf_dfs]


    # plot results
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(
        np.arange(len(ds_means)) - 0.24,
        ds_means, yerr=ds_std, align='center', alpha=0.5, ecolor='black', capsize=10, width=0.4, label='Deepspeed')
    ax.bar(
        np.arange(len(hf_means)) + 0.24,
        hf_means, yerr=hf_std, align='center', alpha=0.5, ecolor='black', capsize=10, width=0.4, label='Huggingface')
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models)
    ax.set_xlabel('Model')
    ax.set_ylabel('Time (s)')
    plt.legend()
    plt.tight_layout()
    plt.title("e2e latency (s), {} tokens, {} world size, {} trials".format(max_tokens, world_size, trials))
    plt.savefig(os.path.join(output_dir, "benchmark.png"))
    print("plot saved to {}".format(os.path.join(output_dir, "benchmark.png")))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='/home/mchorse/benchmarking/output', help="output_directory")
    parser.add_argument("--config", type=str, default='configs/inference_test.yml')
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
         dtype=args.dtype,
         graphs=args.graphs,
         kernel_inject=args.kernel_inject,
         max_tokens=max_tokens,
         local_rank=args.local_rank,
         world_size=world_size,
         trials=trials)

