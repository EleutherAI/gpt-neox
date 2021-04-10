[![GitHub issues](https://img.shields.io/github/issues/EleutherAI/gpt-neox)](https://github.com/EleutherAI/gpt-neox/issues)
[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Weights & Biases monitoring" height=20>](https://wandb.ai/eleutherai/neox)

# GPT-NeoX

This repository records [EleutherAI](www.eleuther.ai)'s work-in-progress for training large scale GPU language models. Our current framework is based on NVIDIA's [Megatron Language Model](https://github.com/NVIDIA/Megatron-LM) and has been augmented with techniques from [DeepSpeed](https://www.deepspeed.ai) as well as some novel optimizations. 

We aim to make this repo a centralized and accessible place to gather techniques for training large scale autoregressive language models, and accelerate research into large scale training. Additionally, we hope to train and open source a 175B parameter GPT3 replication along the way. 

For more info on our progress, please [join our discord](https://discord.com/invite/vtRgjbM) and head to the `#gpt-neo` channel. We're working with cloud compute provider [Coreweave](https://www.coreweave.com/) for training, and hope to release the weights of smaller models as we progress up to 175B parameters.

If you're looking for our TPU codebase, see [GPT-Neo](https://github.com/EleutherAI/gpt-neo).

GPT-NeoX is under active development.

## Features:

### 3D Parallelism 

- GPTNeoX offers full 3D parallelism (data, model and pipeline parallel) using deepspeed, allowing you to scale model training to hundreds of billions of parameters across multiple GPUs.

### Model Structure

- **Positional Encodings:** 

    - Choose between T5 RPE style positional encodings, a learned encoding added to the input (GPT2-style), Sinusoidal positional encoding, and no positional encodings at all (which [recent](https://arxiv.org/abs/1905.04226) [research](https://arxiv.org/abs/2102.11174) has found to even outperform other positional encodings in autoregressive models).

- **Sparsity:** 

    - Deepspeed's sparse attention kernels are supported, but don't work with cuda 11.0+, and require a specific hardware setup (V100s/RTX2080s). add `"sparsity": "all"` to your config to use sparse attention on all layers, or `"sparsity": "interspersed"` to use it every other layer. 

- **Norms:**

    - A [recent Google paper](https://arxiv.org/abs/2102.11972) has shown layernorm may not be the best option for transformer models. 
We offer a choice of layernorm, scalenorm and RMSNorm easily configured by changing a single line in your config file.

### Optimizers

- NeoX supports Adam, CPUAdam, 1-Bit Adam and SM3 optimizers, as well as Deepspeed's [Zero Redundancy Optimizer](https://www.deepspeed.ai/features/#the-zero-redundancy-optimizer).

- **Zero Redundancy Optimizer (ZeRO):** 

    - ZeRO stage 1 works seamlessly with NeoX, while ZeRO stage 2 requires pipeline parallelism be set to 0. We are additionally working on integrating ZeRO 3 into the codebase.
    Turning on ZeRO is as simple as adding one field to your configuration file.

### Straightforward configuration

- Other libraries such as Megatron-LM require you configure them using command line arguments, which can often be difficult to work with and iterate upon. We offer straightforward configuration using .yaml files, which enables you to launch training runs across 100s of GPUs with a single line bash script. 
- Additionally, we hope to make data preparation easier on the user by providing scripts to automatically download and pretokenize a number of large-scale datasets.

## Getting Started

Our codebase relies on [DeeperSpeed](https://github.com/EleutherAI/DeeperSpeed), our fork of the [DeepSpeed](https://github.com/microsoft/DeepSpeed) library with some added changes. 
We strongly recommend using Anaconda, a virtual machine, or some other form of environment isolation before installing from `requirements.txt`. Failure to do so may cause other repositories that rely on DeepSpeed to break.

First make sure you are in an environment with `torch>=1.7.1` installed. Then run `pip install -r requirements.txt`. 
You may need to change the version of `cupy-cudaxxx` to match your machine's cuda version.

Finally, certain features rely on apex, which you can install with the command below:

```bash
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git@e2083df5eb96643c61613b9df48dd4eea6b07690
```

We also host a Docker Image on Dockerhub at `leogao2/gpt-neox`, which enables easy multi-node training.

### Configuration and parameters

GPT-NeoX parameters are defined in a YAML configuration file which is passed to the `deepy.py` launcher - for examples see the `configs` folder. 

For a full list of parameters and documentation see the [configuration readme](configs).

### Datasets

Once you've installed all the requirements and set up your model configuration, the next step is obtaining and preprocessing your dataset. 

For demonstrative purposes we've hosted the Enron Emails corpus and made it available for downloading. Running `python prepare_data.py` will download the tokenizer files and dataset, pretokenize the dataset, and save it into a folder named `./data`.

In the future we will also be adding a single command to preprocess our 800GB language modelling dataset, [The Pile](https://arxiv.org/abs/2101.00027), and all its constituent datasets.

To prepare your own dataset for training, format it as one large jsonl file with each item in the list of dictionaries being a separate document.
The document text should be grouped under one json key, i.e `"text"`. 

Next make sure to download the GPT2 tokenizer vocab, and merge files from the following links:

- Vocab: https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
- Merge: https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt

We plan to integrate HuggingFace's `Tokenizers` library soon to make this process smoother.

You can now pretokenize your data using `tools/preprocess_data.py`.

Usage:

```
preprocess_data.py [-h] --input INPUT [--json-keys JSON_KEYS [JSON_KEYS ...]] [--split-sentences] [--keep-newlines] --tokenizer-type {BertWordPieceLowerCase,BertWordPieceCase,GPT2BPETokenizer} [--vocab-file VOCAB_FILE] [--merge-file MERGE_FILE] [--append-eod]
                          --output-prefix OUTPUT_PREFIX [--dataset-impl {lazy,cached,mmap}] [--workers WORKERS] [--log-interval LOG_INTERVAL]

input data:
  --input INPUT         Path to input JSON
  --json-keys JSON_KEYS [JSON_KEYS ...]
                        space separate listed of keys to extract from json. default = "text".
  --split-sentences     Split documents into sentences.
  --keep-newlines       Keep newlines between sentences when splitting.

tokenizer:
  --tokenizer-type {GPT2BPETokenizer}
                        What type of tokenizer to use.
  --vocab-file VOCAB_FILE
                        Path to the vocab file
  --merge-file MERGE_FILE
                        Path to the BPE merge file (if necessary).
  --append-eod          Append an <eod> token to the end of a document.

output data:
  --output-prefix OUTPUT_PREFIX
                        Path to binary output file without suffix
  --dataset-impl {lazy,cached,mmap}

runtime:
  --workers WORKERS     Number of worker processes to launch
  --log-interval LOG_INTERVAL
                        Interval between progress updates

```

For example:

```bash
python tools/preprocess_data.py \
            --input data/mydataset.jsonl \
            --output-prefix data/mydataset \
            --vocab data/gpt2-vocab.json \
            --dataset-impl mmap \
            --tokenizer-type GPT2BPETokenizer \
            --merge-file gpt2-merges.txt \
            --append-eod
```

You would then run training with the following settings added to your configuration file:

```yaml
  "data-path": "data/mydataset/mydataset",
```

### Training

Training is launched using `deepy.py`, a wrapper around Deepspeed's launcher, which launches the same script in parallel across many GPUs / nodes.

The general usage pattern is:

```bash
./deepy.py [TRAINING_SCRIPT] [path/to/config1.yml] [path/to/config2/yml] ...
```

You can pass in an arbritrary number of configs which will all be merged at runtime.

You can also optionally pass in a config prefix, which will assume all your configs are in the same folder and append that prefix to their path.

Example usage:

```bash
./deepy.py pretrain_gpt2.py -d configs small.yml local_setup.yml
```

This will deploy the `pretrain_gpt2.py` script on all nodes with one process per GPU. The worker nodes and number of GPUs are specified in the `/job/hostfile` file (see [parameter documentation](configs)), or can simply be passed in as the `num_gpus` arg if running on a single node setup.
* Model parameters are defined in the config file `configs/small.yml`.
* Data path parameters are defined in the config file `configs/local_setup.yml`. If you are an EleutherAI member and using the [Kubernetes cluster](kubernetes), the `eleutherai_cluster.yml` config should be instead.

## Monitoring

EleutherAI is currently using [Weights & Biases to record experiments](https://wandb.ai/eleutherai/neox). If you are logged into Weights & Biases on your machine - you can do this by executing `wandb login` - your runs will automatically be recorded. Additionally, set the config parameter `wandb_team` if you would like the run to be added to an organisation/team account.

## Inference

[WIP]

## Eleuther Cluster

We run our experiments on a Kubernetes cluster generously provided by [CoreWeave](https://coreweave.com/). The `/kubernetes/` directory contains code designed to facilitate work on our server. If you are an EleutherAI member, see the [corresponding read-me](kubernetes) for information about how to use our cluster.

## Licensing

This repository hosts code that is part of EleutherAI's GPT-NeoX project. Copyright (c) 2021, EleutherAI contributors (in alphabetical order): Stella Biderman, Sid Black, Eric Hallahan, Josh Levy-Kramer, Michael Pieler, Shivanshu Purohit. Licensed under the Apache License:

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    
        http://www.apache.org/licenses/LICENSE-2.0
    
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

This repository is based off code written by NVIDIA that is licensed under the Apache License, Version 2.0. In accordance with the Apache License, all files that are modifications of code originally written by NVIDIA maintain a NVIDIA copyright header. All files that do not contain such a header are original to EleutherAI contributors. When the NVIDIA code has been modified from its original version, that fact is noted in the copyright header. All derivative works of this repository must preserve these headers under the terms of the Apache License.

For full terms, see the `LICENSE` file. If you have any questions, comments, or concerns about licensing please email us at contact@eleuther.ai.
