[![GitHub issues](https://img.shields.io/github/issues/EleutherAI/gpt-neox)](https://github.com/EleutherAI/gpt-neox/issues)
[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Weights & Biases monitoring" height=20>](https://wandb.ai/eleutherai/neox)

# GPT-NeoX

This repository records [EleutherAI](https://www.eleuther.ai)'s work-in-progress for training large scale GPU language models. Our current framework is based on NVIDIA's [Megatron Language Model](https://github.com/NVIDIA/Megatron-LM) and has been augmented with techniques from [DeepSpeed](https://www.deepspeed.ai) as well as some novel optimizations. 

We aim to make this repo a centralized and accessible place to gather techniques for training large scale autoregressive language models, and accelerate research into large scale training. Additionally, we hope to train and open source a 175B parameter GPT3 replication along the way. 

If you are interested in contributing, please [join our discord](https://discord.gg/zBGx3azzUn) and head to the `#gpt-neo` channel. We're working with cloud compute provider [Coreweave](https://www.coreweave.com/) for training, and hope to release the weights of smaller models as we progress up to 175B parameters.

If you're looking for our TPU codebase, see [GPT-Neo](https://github.com/EleutherAI/gpt-neo).

- [GPT-NeoX](#gpt-neox)
  * [Why GPT-NeoX](#why-gpt-neox)
  * [Quick Start](#quick-start)
  * [Features](#features)
    + [3D Parallelism](#3d-parallelism)
    + [Model Structure](#model-structure)
    + [Optimizers](#optimizers)
    + [High-Precision Training](#high-precision-training)
  * [Datasets](#datasets)
    + [Preconfigured Datasets](#preconfigured-datasets)
    + [Using Custom Data](#using-custom-data)
    + [Using and Training Tokenizers](#using-and-training-tokenizers)
  * [Training and Finetuning](#training-and-finetuning)
  * [Inference](#inference)
  * [Evaluation](#evaluation)
  * [Distilling](#distilling)
  * [Monitoring](#monitoring)
    + [WandB](#wandb)
    + [Tensorboard](#tensorboard)
  * [Placeholder Name](#placeholder-name)
    + [Citing GPT-NeoX](#citing-gpt-neox)
    + [Licensing](#licensing)
    + [Acknowledgements](#acknowledgements)

## Why GPT-NeoX

**Straightforward configuration:** Other libraries such as Megatron-LM require you configure them using command line arguments and global variables, which can often be difficult to work with and iterate upon. We offer straightforward configuration using .yaml files, which enables you to launch training runs across 100s of GPUs with a single line bash script. Additionally, we hope to make data preparation easier on the user by providing scripts to automatically download and pretokenize a number of large-scale datasets.

**Diverse Modeling Options:** We provide a wide collections of options for constructing your model.

**HuggingFace Integration:** Our code is designed to work with the HuggingFace `transformers` library. All models trained using this codebase can be uploaded to a custom HuggingFace class with ease, and all HuggingFace tokenizers and datasets can be used to train models.

**Large Pretrained Models:** We offer several large, pretrained models to iterate on. For people who are unable to train billion parameter scale models themselves, this framework allows you to easily interact with models that we have released.

## Quick Start

**Google Colab**

Coming soon: a colab notebook for trying out the model.

**Warning:** Our codebase relies on [DeeperSpeed](https://github.com/EleutherAI/DeeperSpeed), our fork of the [DeepSpeed](https://github.com/microsoft/DeepSpeed) library with some added changes. We strongly recommend using Anaconda, a [virtual machine](#using-docker), or some other form of environment isolation before installing from `requirements/requirements.txt`. Failure to do so may cause other repositories that rely on DeepSpeed to break.

### Local Environment
First make sure you are in an environment with Python 3.8 or later and `torch>=1.8` installed. Then run `pip install -r requirements/requirements.txt`. 
You may need to change the version of `cupy-cudaxxx` to match your machine's cuda version.

nvidia's apex is an optional extra (used only for FusedAdam, which may offer some performance improvement):

```bash
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git@e2083df5eb96643c61613b9df48dd4eea6b07690
```

### Using Docker
It may be preferable to use this code through a Docker image based on the Dockerfile we provide, which takes care of most of the installation process. To use this option, first build an image named `gptneox-image` from the repository root directory using:
```
docker build -t gptneox-image -f Dockerfile .
```
You can then run a container based on this image. For instance, the below snippet mounts the cloned repository (`gpt-neox`) directory to `/gpt-neox` in the container and uses [NVidia-Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to make four GPUs (numbers 0-3) accessible to the container. Note the `--shm-size=1g --ulimit memlock=-1` flags: in our epxerience, without these the container allocates far too little shared memory for training.
```
nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 --shm-size=1g --ulimit memlock=-1 --mount type=bind,src=$PWD,dst=/gpt-neox gptneox-image
```
After navigating to `/gpt-neox` within this container, you should then be able to run the code as described below. Upon the first run, you may be prompted to "install the fused kernels" with a provided command. Additionally, extra requirements, such as TensorBoard, are not installed by default. You are strongly encouraged (and often required) to run every command as root (using `sudo`).

We also host a Docker Image on Dockerhub at `leogao2/gpt-neox`, which enables easy multi-node training.

### Running the Code
Once you've installed all the requirements and set up your model configuration, the next step is obtaining and preprocessing your dataset. We provide a data processing library that is easily interfaced with via the function `prepare_data.py`. Calling `python prepare_data.py enron -t CharLevelTokenizer -d ./data/` will download the dataset `enron`, tokenize it with a character-level tokenizer, and save the results to `./data/`. 

GPT-NeoX parameters are defined in a YAML configuration file which is passed to the `deepy.py` launcher. We provide baseline examples for the models found in the paper [Language Models are Few Shot Learners](https://arxiv.org/abs/2005.14165). Configs such as file locations that are dependant on your particular system go in `local_setup.yml`. We have filled it out with some placeholder examples, but you will need to update this for your system.

All functionality follows the pattern `./deepy.py main_function.py -d configs small.yml local_setup.yml`
We currently offer four main functions:
1. `pretrain_gpt2.py` is used for training and finetuning models.
2. `eval_tasks/run.py` is used to evaluate a trained model using the evaluation harness.
3. `text_gen_gpt2.py` is used to sample text from a trained model.

For now, run `./deepy.py pretrain_gpt2.py -d configs small.yml local_setup.yml` to begin training a model and complete this tutorial.

## Features

GPT-NeoX offers a wide variety of state-of-the-art and bespoke features 

### 3D Parallelism 

- GPTNeoX offers full 3D parallelism (data, model and pipeline parallel) using DeepSpeed, allowing you to scale model training to hundreds of billions of parameters across multiple GPUs.

### Model Structure

- **Positional Encodings:** 

    - Choose between T5-style relative positional encodings, learned encoding added to the input (GPT2-style), sinusoidal positional encoding, [rotary positional encodings](https://arxiv.org/abs/2104.09864), and no positional encodings at all (which [recent](https://arxiv.org/abs/1905.04226) [research](https://arxiv.org/abs/2102.11174) has found to be competetive with other positional encodings in autoregressive models). Use the `pos-emb` field to select a positional encoding.

- **Sparsity:** 

    - Deepspeed's sparse attention kernels are supported, but don't work with cuda 11.0+, and require a specific hardware setup (V100s/RTX2080s/A100s). Add `"sparsity": "all"` to your config file to use sparse attention on all layers, or `"sparsity": "interspersed"` to use it every other layer. To use sparsity, first run `pip install requirements/requirements-sparseattention.txt` to install triton.

- **Norms:**

    - We offer a choice of layernorm, scalenorm, RMSNorm, and a custom layernorm kernel. Use the `norm` field to select a normalization.

### Optimizers

- NeoX supports Adam, CPUAdam, 1-Bit Adam, SM3 and madgrad_wd optimizers, as well as Deepspeed's [Zero Redundancy Optimizer](https://www.deepspeed.ai/features/#the-zero-redundancy-optimizer). Use the `optimizer` and (if applicable) `zero_optimization` fields to configure your optimizer.

- **Zero Redundancy Optimizer (ZeRO):** 

    - ZeRO stage 1 works seamlessly with NeoX, while ZeRO stage 2 requires pipeline parallelism be set to 0. We are additionally working on integrating ZeRO 3 into the codebase.
    Turning on ZeRO is as simple as adding one field to your configuration file.

### High-Precision Training

 - Choose between `fp16`, `bf16`, and `fp32` operations to get the most performance out of your avaliable compute. Use the `precision` field to configure your precision settings, by adding `"type": "bfloat16"` in the config.
 - Due to a known issue with `PyTorch`, `bf16` models require doing the all-reduce operation in `fp32`. If you have a patch for this problem, you can turn off the default`"fp32_allreduce": True`.
 - Additionally, you have to run `python /megatron/fused_kernels/setup.py install` (assuming you're inside `gpt-neox/`) to be able to use bf16 (may require root access).

## Datasets


### Preconfigured Datasets

For demonstrative purposes we've hosted the Enron Emails corpus and made it available for downloading. Running `python prepare_data.py` will download the tokenizer files and dataset, pretokenize the dataset, and save it into a folder named `./data`.

In the future we will also be adding a single command to preprocess our 800GB language modelling dataset, [The Pile](https://arxiv.org/abs/2101.00027), and all its constituent datasets.

To prepare your own dataset for training, format it as one large jsonl file with each item in the list of dictionaries being a separate document.
The document text should be grouped under one json key, i.e `"text"`. 

Next make sure to download the GPT2 tokenizer vocab, and merge files from the following links:

- Vocab: https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
- Merge: https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt

### Using Custom Data

### Using and Training Tokenizers

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

## Training and Finetuning

Training is launched using `deepy.py`, a wrapper around Deepspeed's launcher, which launches the same script in parallel across many GPUs / nodes.

The general usage pattern is:

```bash
./deepy.py [TRAINING_SCRIPT] [path/to/config1.yml] [path/to/config2.yml] ...
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


## Inference

[WIP]

## Evaluation

GPT-NeoX supports evaluation on downstream tasks through the [language model evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness).

To evaluate a trained model on the evaluation harness, use `./deepy.py evaluate.py configs/your_config.yml`

## Distilling

Coming soon! Check out the `distill-gpt-neox` branch to try distilling a model.

## Monitoring

### WandB

EleutherAI is currently using [Weights & Biases to record experiments](https://wandb.ai/eleutherai/neox). If you are logged into Weights & Biases on your machine - you can do this by executing `wandb login` - your runs will automatically be recorded. Additionally, set the config parameter `wandb_team` if you would like the run to be added to an organisation/team account.

### Tensorboard

We also support using Tensorboard via the `tensorboard-dir` argument. To use tensorboard, install the optional packages found at `requirements/requirements-tensorboard.txt`

## Placeholder Name

### Citing GPT-NeoX

If you have found GPT-Neo helpful in your work, you can cite this repository as

```
@software{gpt-neo,
  author = {Andonian, Alex and Biderman, Stella and Black, Sid and Gali, Preetham and Gao, Leo and Hallahan, Eric and Levy-Kramer, Josh and Leahy, Connor and Nestler, Lucas and Parker, Kip and Pieler, Michael and Purohit, Shivanshu and Songz, Tri and Wang, Phil and Weinbach, Samuel},
  title = {{GPT-NeoX}: Large Scale Autoregressive Language Modeling in PyTorch},
  url = {http://github.com/eleutherai/gpt-neox},
  year = {2021}
}
```

In the above bibtex entry, names are in alphabetical order, and the year corresponds to the project's open-source release.

### Licensing

This repository hosts code that is part of EleutherAI's GPT-NeoX project. Copyright (c) 2021, EleutherAI contributors (in alphabetical order): Alex Andonian, Stella Biderman, Sid Black, Preetham Gali, Leo Gao, Eric Hallahan, Josh Levy-Kramer, Connor Leahy, Lucas Nestler, Kip Parker, Michael Pieler, Shivanshu Purohit, Tri Songz, Phil Wang, Samuel Weinbach. Licensed under the Apache License:

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

### Acknowledgements

We run our experiments on a Kubernetes cluster generously provided by [CoreWeave](https://coreweave.com/).

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

