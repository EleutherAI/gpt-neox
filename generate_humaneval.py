#!/usr/bin/env python
# Copyright (c) 2021 Josh Levy-Kramer <josh@levykramer.co.uk>. All rights reserved.
# This file is based on code by the authors denoted below and has been modified from its original version.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from megatron.utils import print_rank_0, setup_for_inference_or_eval

from megatron.text_generation_utils import (
    generate_samples_from_fim_prompt,
)

from human_eval_infilling.data import write_jsonl, read_problems


def main():
    """
    Generate text/sample model
    """
    model, neox_args = setup_for_inference_or_eval(use_cache=True)
    if neox_args.recompute:
        model.module.inference_mode(
            use_cache=False
        )  # don't use kv cache if recomputing

    print_rank_0(
        f"Generating samples and saving results to {neox_args.sample_output_file}"
    )

    # create a list of prompts in <SUF> Suffix <PRE> Prefix <Mid> form, as strings?
    problems = read_problems(benchmark_name="single-line")
    # where each problem is repeated K times
    K = 100
    inputs = [problem for problem in problems for _ in range(K)]

    generated_texts = generate_samples_from_fim_prompt(
        neox_args=neox_args,
        model=model,
        text=inputs,
        eos_token_id=None,
        maximum_tokens=64,
        recompute=neox_args.recompute,
        temperature=neox_args.temperature,
        top_k=neox_args.top_k,
        top_p=neox_args.top_p,
        stop_tokens=None,
    )

    completions = [
        {"task_id": completion["task_id"], 
        "completion": completion["text"]} 
        for completion in generated_texts
    ]

    write_jsonl("./samples.jsonl", completions)

if __name__ == "__main__":
    main()
