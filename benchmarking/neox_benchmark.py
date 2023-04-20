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

from megatron.text_generation_utils import generate_samples_from_prompt
from megatron.utils import print_rank_0, setup_for_inference_or_eval


PYTHIA_TO_OLD_SUFFIXES = {
    "70m": "19M",
    "160m": "125M",
    "410m": "350M",
    "1b": "800M",
    "1.4b": "1-3B",
    "2.8b": "2.7B",
    "6.9b": "6-7B",
    "12b": "13B",
    "20b": "20B"}


def main():
    model, neox_args = setup_for_inference_or_eval(use_cache=True)
    max_tokens = 10
    print_rank_0("Finished loading model")

    prompts = ["DeepSpeed is" for x in range(100)]

    generated_texts = generate_samples_from_prompt(
        neox_args=neox_args,
        model=model,
        text=prompts,
        eos_token_id=0,
        maximum_tokens=10,
        recompute=neox_args.recompute,
        temperature=neox_args.temperature,
        top_k=neox_args.top_k,
        top_p=neox_args.top_p,
    )

    print(generated_texts)
    times = [x["duration_seconds"] for x in generated_texts]

    for_dataframe = np.vstack((times, list(map(lambda t: t / (max_tokens - 3), times)))).T
    columns = ["(e2e) latency", "(e2e) per token latency"]

    df = pd.DataFrame(
        for_dataframe,
        columns = columns)


    # save dataframe to CSV inside the directory for world_size
    # if local_rank == 0:

    # neox_dir = os.path.join(output_dir, "neox")
    # max_tokens_dir = os.path.join(neox_dir, "max_tokens_{}".format(max_tokens))
    # world_size_dir = os.path.join(max_tokens_dir, "world_size_{}".format(world_size))

    # os.makedirs(world_size_dir, exist_ok=True)

    # fname = os.path.join(world_size_dir,
    #                     "{}_fp16_benchmark.csv".format(model.split('/')[-1]))
    
    # print("saving benchmark to {}".format(fname))
    # df.to_csv(fname, index=False)
    print(df)
    return df


if __name__ == "__main__":
    main()

