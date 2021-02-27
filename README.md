[![GitHub issues](https://img.shields.io/github/issues/EleutherAI/gpt-neox)](https://github.com/EleutherAI/gpt-neox/issues)
[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Weights & Biases monitoring" height=20>](https://wandb.ai/eleutherai/neox)

# GPT-NeoX

This repository records [EleutherAI](www.eleuther.ai)'s work-in-progress for training large scale GPU language models. Our current framework is based on NVIDIA's [Megatron Language Model](https://github.com/NVIDIA/Megatron-LM) and has been augmented with techniques from [DeepSpeed](https://www.deepspeed.ai) as well as some novel optimizations. 

If you are looking for our TPU codebase, see [GPT-Neo](https://github.com/EleutherAI/gpt-neo).

GPT-NeoX is under active development and rough around the edges. GPT-NeoX is a complicated beast that will take time and patients to work on any specific environment.

## Getting Started

Our codebase relies on [DeeperSpeed](https://github.com/EleutherAI/DeeperSpeed), a custom modification to the [DeepSpeed](https://github.com/microsoft/DeepSpeed) library. We strongly recommend using Anaconda, a virtual environment, or some other form of environment isolation before installing from `requirements.txt`. Failure to do so may cause other repositories that rely on DeepSpeed to break.

## Datasets

Once you've installed `requirements.txt`, the next step is obtaining and processing data. For demonstrative purposes we have hosted the Enron Emails corpus and made it available for downloading. Running `python prepare_data.py` will download and process the dataset for language modeling. To use your own data, extend the `DataDownloader` class in `tools/corpa.py`and register the new class in the `DATA_DOWNLOADERS` dict. Once this is done, you can add `prepare_dataset(dataset_name)` to `process_data.py` to load your data.

TO DO: Make a table showing the datasets currently available for download. List the name, size on disk (compressed), actual size, and number of tokens.

### Training

If you are already familiar with training models using DeepSpeed, you can use the exact same API to train our models.

## Features

### Model Structure

**Positional Encoding:** Currently we only support sinesoidal positional encodings.

**Sparsity:** Sparse attention kernels are supported, but they require model parallelism to be turned off. This is subject to change with updates in Deepspeed

### Optimizers

**Zero Redundnacy Optimizer (ZeRO):** ZeRO stage 1 works seamlessly with NeoX, while ZeRO stage 2 does not, as it requires disabling pipeline parallelsm due to conflicts with gradient checkpointing among the two features. 

**ZeRO-Offloding:** ZeRO-offloading requires ZeRO stage 2, hence is not supported.

**1-Bit Adam:**

### Memory Optimizations

**Data Parallel:** Data parallelism is a ubiquitous technique in deep learning in which each input batch of training data is split among the data parallel workers. It is integrated into NeoX

**Model Parallel:** Model Parallelism is a broad class of techniques that partitions the individual layers of the model across workers. Model Parallelism is built into NeoX as it is a part of [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

**Pipeline Parallel:** Pipeline parallelism divides the layers of the model into stages that can be processed in parallel. It is integrated into deepspeed itself.

**Mixed Precision Training:** Mixed precision training computes some operations in FP16 while some others in FP32, such as computing the forward pass and the gradient in fp16 and updating the weights in fp32. Mixed precision training is integrated into deepspeed as well.

## Monitoring

EleutherAI is currently using [Weights & Biases to record experiments](https://wandb.ai/eleutherai/neox). If you are logged into Weights & Biases on your machine - you can do this by executing `wandb login` - your runs will automatically be recorded. Additionally, set the environment variable `WANDB_TEAM` if you would like the run to be added to a organisation/team account.

## Eleuther Cluster

We run our experiments on a Kubernetes cluster generously provided by [CoreWeave](https://coreweave.com/). The `/kubernetes/` directory contains code designed to facilitate work on our server. If you are an EleutherAI member, see the [corresponding read-me](kubernetes/README.md) for information about how to use our cluster.

## Licensing

This repository hosts code that is part of EleutherAI's GPT-NeoX project. Copyright 2021 Stella Biderman, Sid Black, Leo Gao, Josh Levy-Kramer, and Shivanshu Purohit.

    GPT-NeoX is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <https://www.gnu.org/licenses/>.

This repository is based off code written by NVIDIA that is licensed under the Apache License, Version 2.0. In accordance with the Apache License, all files that are modifications of code originally written by NIVIDIA maintain a NVIDIA copyright header. All files that do not contain such a header are original to EleutherAI. When the NVIDIA code has been modified from its original version, that fact is noted in the copyright header. All derivative works of this repository must preserve these headers under the terms of the Apache License.

For full terms, see the `LICENSE` file. If you have any questions, comments, or concerns about licensing please email us at contact@eleuther.ai.
