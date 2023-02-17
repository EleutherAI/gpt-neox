[![GitHub issues](https://img.shields.io/github/issues/EleutherAI/gpt-neox)](https://github.com/EleutherAI/gpt-neox/issues)
[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Weights & Biases monitoring" height=20>](https://wandb.ai/eleutherai/neox)

# GPT-NeoX

This repository records [EleutherAI](https://www.eleuther.ai)'s library for training large-scale language models on GPUs. Our current framework is based on NVIDIA's [Megatron Language Model](https://github.com/NVIDIA/Megatron-LM) and has been augmented with techniques from [DeepSpeed](https://www.deepspeed.ai) as well as some novel optimizations. We aim to make this repo a centralized and accessible place to gather techniques for training large-scale autoregressive language models, and accelerate research into large-scale training.

For those looking for a TPU-centric codebase, we recommend [Mesh Transformer JAX](https://github.com/kingoflolz/mesh-transformer-jax).

**If you are not looking to train models with billions of parameters from scratch, this is likely the wrong library to use. For generic inference needs, we recommend you use the Hugging Face `transformers` library instead which supports GPT-NeoX models.**

# Contents

- [Pretrained Models](#pretrained-models)
  * [GPT-NeoX-20B](#gpt-neox-20b)
  * [Pythia](#pythia)
  * [Polyglot](#polyglot)
  * [Fill-in-the-Middle](#fill-in-the-middle)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Datasets](#datasets)
  * [Preconfigured Datasets](#preconfigured-datasets)
  * [Using Custom Data](#using-custom-data)
- [Training and Finetuning](#training-and-finetuning)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Exporting to Hugging Face](#exporting-to-hugging-face)
- [Monitoring](#monitoring)
  * [Weights & Biases](#wandb)
  * [TensorBoard](#tensorboard)
- [Administrative Notes](#administrative-notes)
  * [Citing GPT-NeoX](#citing-gpt-neox)
  * [Licensing](#licensing)
  * [Publications](#publications)
  * [Acknowledgements](#acknowledgements)

# Pretrained Models

## GPT-NeoX-20B

GPT-NeoX-20B is a 20 billion parameter autoregressive language model trained on [the Pile](https://arxiv.org/abs/2101.00027). Technical details about GPT-NeoX-20B can be found in [the associated paper](https://arxiv.org/abs/2204.06745). The configuration file for this model is both available at [`./configs/20B.yml`](./configs/20B.yml) and included in the download links below.

### Download Links

[Slim weights](https://the-eye.eu/public/AI/models/GPT-NeoX-20B/slim_weights/) - (No optimizer states, for inference or finetuning, 39GB)

To download from the command line to a folder named `20B_checkpoints`, use the following command:

```bash
wget --cut-dirs=5 -nH -r --no-parent --reject "index.html*" https://the-eye.eu/public/AI/models/GPT-NeoX-20B/slim_weights/ -P 20B_checkpoints
```

[Full weights](https://the-eye.eu/public/AI/models/GPT-NeoX-20B/full_weights/) - (Including optimizer states, 268GB)

To download from the command line to a folder named `20B_checkpoints`, use the following command:

```bash
wget --cut-dirs=5 -nH -r --no-parent --reject "index.html*" https://the-eye.eu/public/AI/models/GPT-NeoX-20B/full_weights/ -P 20B_checkpoints
```

Weights can be alternatively be downloaded using a BitTorrent client. Torrent files can be downloaded here: [slim weights](https://the-eye.eu/public/AI/models/GPT-NeoX-20B/slim_weights.torrent), [full weights](https://the-eye.eu/public/AI/models/GPT-NeoX-20B/full_weights.torrent).

We additionally have 150 checkpoints saved throughout training, one every 1,000 steps. We are working on figuring out how to best serve these at scale, but in the meanwhile people interested in working with the partially trained checkpoints can email us at contact@eleuther.ai to arrange access.

## Pythia

The Pythia Scaling Suite is a suite of models ranging from 19M parameters to 13B parameters trained on [the Pile](pile.eleuther.ai) intended to promote research on interpretability and training dynamics of large language models. Further details about the project and links to the models can be found [here](https://github.com/EleutherAI/pythia).

## Polyglot

The Polyglot Project is an effort to train powerful non-English pretrained language models to promote the accessibility of this technology to researchers outside the dominant powerhouses of machine learning. EleutherAI has trained and released 1.3B, 3.8B, and 5.8B parameter Korean language models, the largest of which outpreforms all other publicly available language models on Korean language tasks. Further details about the project and links to the models can be found [here](https://github.com/EleutherAI/polyglot).

## Fill-in-the-Middle

EleutherAI's [Carper lab](https://www.carper.ai) has also used this codebase to train models using FIM (fill-in-the-middle), a data transformation proposed in [Bavarian et al. 2022](https://arxiv.org/abs/2207.14255) with a similar technique also used by [Fried et al.](https://arxiv.org/abs/2204.05999) and [Aghajanyan et al. 2022](https://arxiv.org/abs/2201.07520), to enable typically autoregressive left-to-right language models to perform text infilling conditioned on both "left" and "right" context. A 1.3B parameter model trained on [the Pile](pile.eleuther.ai) is available [here](https://huggingface.co/CarperAI/FIM-NeoX-1.3B), with further experiments and and models forthcoming.

# Quick Start

## Environment and Dependencies

### Host Setup

First make sure you are in an environment with Python 3.8 with an appropriate version of PyTorch 1.8 or later installed. **Note:** Some of the libraries that GPT-NeoX depends on have not been updated to be compatible with Python 3.10+. Python 3.9 appears to work, but this codebase has been developed and tested for Python 3.8.

To install the remaining basic dependencies, run:

```bash
pip install -r requirements/requirements.txt
python ./megatron/fused_kernels/setup.py install # optional if not using fused kernels
```

from the repository root.

<aside>

**Warning:** Our codebase relies on [DeeperSpeed](https://github.com/EleutherAI/DeeperSpeed), our fork of the [DeepSpeed](https://github.com/microsoft/DeepSpeed) library with some added changes. We strongly recommend using Anaconda, a virtual machine, or some other form of environment isolation before continuing. Failure to do so may cause other repositories that rely on DeepSpeed to break.

</aside>


### Flash Attention

To use [Flash-Attention](https://github.com/HazyResearch/flash-attention), install the additional dependencies in  `./requirements/requirements-flashattention.txt` and set the attention type in your configuration accordingly (see [configs](./configs/)). This can provide significant speed-ups over regular attention on certain GPU architectures, including Ampere GPUs (such as A100s); see the repository for more details.


### Containerized Setup

We also provide a Dockerfile if you prefer to run NeoX in a container. To use this option, first build an image named `gpt-neox` from the repository root directory with `docker build -t gpt-neox -f Dockerfile .`. We also host pre-built images on Docker Hub at `leogao2/gpt-neox`.

You can then run a container based on this image. For instance, the below snippet mounts the cloned repository (`gpt-neox`) directory to `/gpt-neox` in the container and uses [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to make four GPUs (numbers 0-3) accessible to the container. [As noted by the NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#sharing-data), both `--shm-size=1g` and `--ulimit memlock=-1` are important to prevent Docker from allocating too little shared memory.
```
nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 --shm-size=1g --ulimit memlock=-1 --mount type=bind,src=$PWD,dst=/gpt-neox gpt-neox
```

## Usage

All functionality (inference included), should be launched using `deepy.py`, a wrapper around the `deepspeed` launcher.

We currently offer three main functions:
1. `train.py` is used for training and finetuning models.
2. `evaluate.py` is used to evaluate a trained model using the [language model evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness).
3. `generate.py` is used to sample text from a trained model.

which can be launched with:

```bash
./deepy.py [script.py] [./path/to/config_1.yml] [./path/to/config_2.yml] ... [./path/to/config_n.yml]
```

E.G To generate text unconditionally with the GPT-NeoX-20B model, you can use the following:
```bash
./deepy.py generate.py ./configs/20B.yml
```

Or optionally pass in a text file (e.g `prompt.txt`) to use as the prompt, which should be a plain `.txt` file with each prompt separated by newline characters, also passing in the path to an output file.

```bash
./deepy.py generate.py ./configs/20B.yml -i prompt.txt -o sample_outputs.txt
```

To reproduce our evaluation numbers on, for example, TriviaQA and PIQA use:

```bash
./deepy.py evaluate.py ./configs/20B.yml --eval_tasks triviaqa piqa
```

You can add an arbitrary list of evaluation tasks here, for details of all tasks available, see [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

For more details on each entry point, see the [Training and Finetuning](#training-and-finetuning), [Inference](#inference) and [Evaluation](#evaluation)
# Configuration

GPT-NeoX parameters are defined in a YAML configuration file which is passed to the deepy.py launcher. We have provided some example .yaml files in [configs](./configs/), including one for GPT-NeoX-20B, and example configuration files for other model sizes.

These files are generally complete, but non-optimal. For example, depending on your specific GPU configuration, you may need to change some settings such as `pipe-parallel-size`, `model-parallel-size` to increase or decrease the degree of parallelisation, `train_micro_batch_size_per_gpu` or `gradient-accumulation-steps` to modify batch size related settings, or the `zero_optimization` dict to modify how optimizer states are parallelised across workers.

For a more detailed guide to all the features available and how to configure them, see [the configuration README](configs/README.md), and for documentation of every possible argument, see [configs/neox_arguments.md](configs/neox_arguments.md).

# Datasets

## Preconfigured Datasets

Several preconfigured datasets are available, including most components from [the Pile](https://arxiv.org/abs/2101.00027), as well as the Pile train set itself, for straightforward tokenization using the `prepare_data.py` entry point.

E.G, to download and tokenize the Enron emails corpus with the GPT2 Tokenizer, saving them to `./data` you can run:

```
python prepare_data.py -d ./data
```

or with the GPT-NeoX-20B tokenizer (assuming you have it saved at `./20B_checkpoints/20B_tokenizer.json`):

```
python prepare_data.py -d ./data -t HFTokenizer --vocab-file ./20B_checkpoints/20B_tokenizer.json
```

The tokenized data will be saved out to two files: `[data-dir]/[dataset-name]/[dataset-name]_text_document.bin`and `[data-dir]/[dataset-name]/[dataset-name]_text_document.idx`. You will need to add the prefix that both these files share to your training configuration file under the `data-path` field. E.G:

```yaml
  "data-path": "./data/enron/enron_text_document",
```

## Using Custom Data

To prepare your own dataset for training with custom data, format it as one large [jsonl](https://jsonlines.org/)-formatted file with each item in the list of dictionaries being a separate document. The document text should be grouped under one JSON key, i.e `"text"`. Any auxiliary data stored in other fields will not be used.

Next make sure to download the GPT2 tokenizer vocab, and merge files from the following links:

- Vocab: https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
- Merge: https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt

Or use the 20B tokenizer (for which only a single Vocab file is needed):

- Vocab: https://the-eye.eu/public/AI/models/GPT-NeoX-20B/slim_weights/20B_tokenizer.json

(alternatively, you can provide any tokenizer file that can be loaded by Hugging Face's tokenizers library with the `Tokenizer.from_pretrained()` command)

You can now pretokenize your data using `tools/preprocess_data.py`, the arguments for which are detailed below:

```
usage: preprocess_data.py [-h] --input INPUT [--jsonl-keys JSONL_KEYS [JSONL_KEYS ...]] [--num-docs NUM_DOCS] --tokenizer-type {HFGPT2Tokenizer,HFTokenizer,GPT2BPETokenizer,CharLevelTokenizer} [--vocab-file VOCAB_FILE] [--merge-file MERGE_FILE] [--append-eod] [--ftfy] --output-prefix OUTPUT_PREFIX
                          [--dataset-impl {lazy,cached,mmap}] [--workers WORKERS] [--log-interval LOG_INTERVAL]

optional arguments:
  -h, --help            show this help message and exit

input data:
  --input INPUT         Path to input jsonl files or lmd archive(s) - if using multiple archives, put them in a comma separated list
  --jsonl-keys JSONL_KEYS [JSONL_KEYS ...]
                        space separate listed of keys to extract from jsonl. Defa
  --num-docs NUM_DOCS   Optional: Number of documents in the input data (if known) for an accurate progress bar.

tokenizer:
  --tokenizer-type {HFGPT2Tokenizer,HFTokenizer,GPT2BPETokenizer,CharLevelTokenizer}
                        What type of tokenizer to use.
  --vocab-file VOCAB_FILE
                        Path to the vocab file
  --merge-file MERGE_FILE
                        Path to the BPE merge file (if necessary).
  --append-eod          Append an <eod> token to the end of a document.
  --ftfy                Use ftfy to clean text

output data:
  --output-prefix OUTPUT_PREFIX
                        Path to binary output file without suffix
  --dataset-impl {lazy,cached,mmap}
                        Dataset implementation to use. Default: mmap

runtime:
  --workers WORKERS     Number of worker processes to launch
  --log-interval LOG_INTERVAL
                        Interval between progress updates

```

For example:

```bash
python tools/preprocess_data.py \
            --input ./data/mydataset.jsonl.zst \
            --output-prefix ./data/mydataset \
            --vocab ./data/gpt2-vocab.json \
            --merge-file gpt2-merges.txt \
            --dataset-impl mmap \
            --tokenizer-type GPT2BPETokenizer \
            --append-eod
```

You would then run training with the following settings added to your configuration file:

```yaml
  "data-path": "data/mydataset/mydataset",
```

# Training and Finetuning

Training is launched using `deepy.py`, a wrapper around DeepSpeed's launcher, which launches the same script in parallel across many GPUs / nodes.

The general usage pattern is:

```bash
python ./deepy.py train.py [path/to/config1.yml] [path/to/config2.yml] ...
```

You can pass in an arbitrary number of configs which will all be merged at runtime.

You can also optionally pass in a config prefix, which will assume all your configs are in the same folder and append that prefix to their path.

E.G:

```bash
python ./deepy.py train.py -d configs small.yml local_setup.yml
```

This will deploy the `train.py` script on all nodes with one process per GPU. The worker nodes and number of GPUs are specified in the `/job/hostfile` file (see [parameter documentation](configs/README.md)), or can simply be passed in as the `num_gpus` arg if running on a single node setup.

Although this is not strictly necessary, we find it useful to define the model parameters in one config file (e.g `configs/small.yml`) and the data path parameters in another (e.g `configs/local_setup.yml`).

# Inference

**For most uses we recommend deploying models trained using the GPT-NeoX library via the Hugging Face Transformers library which is better optimized for inference.**

We support three types of generation from a pretrained model:
1. Unconditional generation
2. Conditional generation based on an input read from a file
3. Interactive generation, which allows for multiple rounds of back-and-forth between a user and the language model via a command line interface

All three types of text generation can be launched via `python ./deepy.py generate.py -d configs small.yml local_setup.yml text_generation.yml` with the appropriate values set in `configs/text_generation.yml`.

# Evaluation

GPT-NeoX supports evaluation on downstream tasks through the [language model evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness).

To evaluate a trained model on the evaluation harness, simply run:

```bash
python ./deepy.py evaluate.py -d configs your_configs.yml --eval_tasks task1 task2 ... taskn
```

where `--eval_tasks` is a list of evaluation tasks followed by spaces, e.g `--eval_tasks lambada hellaswag piqa sciq`. For details of all tasks available, refer to the [lm-evaluation-harness repo](https://github.com/EleutherAI/lm-evaluation-harness).

# Exporting to Hugging Face

GPT-NeoX is optimized heavily for training only, and GPT-NeoX model checkpoints are not compatible out of the box with other deep learning libraries. To make models easily loadable and shareable with end users, and for further exporting to various other frameworks, GPT-NeoX supports checkpoint conversion to the [Hugging Face Transformers](https://arxiv.org/abs/1910.03771) GPTNeoXModel format.

To convert a NeoX checkpoint to Hugging Face-loadable format, run:
```bash
python ./tools/convert_to_hf.py --input_dir /path/to/model/global_stepXXX --config_file your_config.yml --output_dir hf_model/save/location
```
Then to upload a model to [the Hugging Face Hub](https://huggingface.co/), run:
```
huggingface-cli login
python ./tools/upload.py
```
and input the requested information, including HF hub user token.

Note, however, that this compatibility is not one-to-one, and only certain configurations from GPT-NeoX are supported in the Hugging Face GPTNeoXModel class. Advanced features such as alternative positional embeddings may require new Transformers modeling code and new conversion script tweaks.

# Monitoring

In addition to storing logs locally, we provide built-in support for two popular experiment monitoring frameworks: [Weights & Biases](https://wandb.ai/site) and [TensorBoard](https://www.tensorflow.org/tensorboard/)

<h2 id="wandb">Weights & Biases</h2>

EleutherAI is currently using [Weights & Biases to record our experiments](https://wandb.ai/eleutherai/neox). If you are logged into Weights & Biases on your machine&mdash;you can do this by executing `wandb login`&mdash;your runs will automatically be recorded. There are two optional fields associated with Weights & Biases: <code><var>wandb_group</var></code> allows you to name the run group and <code><var>wandb_team</var></code> allows you to assign your runs to an organization or team account.

## TensorBoard

We also support using TensorBoard via the <code><var>tensorboard-dir</var></code> field. Dependencies required for TensorBoard monitoring can be found in and installed from  `./requirements/requirements-tensorboard.txt`.

# Running on multi-node

If you need to supply a hostfile for use with the MPI-based DeepSpeed launcher, you can set the environment variable `DLTS_HOSTFILE` to point to the hostfile.

# Administrative Notes

## Citing GPT-NeoX

If you have found the GPT-NeoX library helpful in your work, you can cite this repository as

```bibtex
@software{gpt-neox-library,
  title = {{GPT-NeoX: Large Scale Autoregressive Language Modeling in PyTorch}},
  author = {Andonian, Alex and Anthony, Quentin and Biderman, Stella and Black, Sid and Gali, Preetham and Gao, Leo and Hallahan, Eric and Levy-Kramer, Josh and Leahy, Connor and Nestler, Lucas and Parker, Kip and Pieler, Michael and Purohit, Shivanshu and Songz, Tri and Phil, Wang and Weinbach, Samuel},
  url = {https://www.github.com/eleutherai/gpt-neox},
  doi = {10.5281/zenodo.5879544},
  month = {8},
  year = {2021},
  version = {0.0.1},
}
```

To cite our 20 billion parameter model, please use

```bibtex
@inproceedings{gpt-neox-20b,
  title={{GPT-NeoX-20B}: An Open-Source Autoregressive Language Model},
  author={Black, Sid and Biderman, Stella and Hallahan, Eric and Anthony, Quentin and Gao, Leo and Golding, Laurence and He, Horace and Leahy, Connor and McDonell, Kyle and Phang, Jason and Pieler, Michael and Prashanth, USVSN Sai and Purohit, Shivanshu and Reynolds, Laria and Tow, Jonathan and Wang, Ben and Weinbach, Samuel},
  booktitle={Proceedings of the ACL Workshop on Challenges \& Perspectives in Creating Large Language Models},
  url={https://arxiv.org/abs/2204.06745},
  year={2022}
}
```

Citation instructions for other pretrained models can be found [in the appropriate repository](#pretrained-models).

## Licensing

This repository hosts code that is part of EleutherAI's GPT-NeoX project. Copyright (c) 2021, EleutherAI. Licensed under the Apache License:

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

This repository is based off code written by NVIDIA that is licensed under the Apache License, Version 2.0. In accordance with the Apache License, all files that are modifications of code originally written by NVIDIA maintain a NVIDIA copyright header. All files that do not contain such a header are the exclusive copyright of EleutherAI. When the NVIDIA code has been modified from its original version, that fact is noted in the copyright header. All derivative works of this repository must preserve these headers under the terms of the Apache License.

This repository also contains code written by a number of other authors. Such contributions are marked and the relevant licensing is included where appropriate.

For full terms, see the `LICENSE` file. If you have any questions, comments, or concerns about licensing please email us at contact@eleuther.ai.

## Publications

The following publications have come out of this project:

 - Black, Biderman, Hallahan, Anthony, Gao, Golding, He, Leahy, McDonell, Phang, Pieler, Prashanth, Purohit, Reynolds, Tow, Wang, and Weinbach. "[GPT-NeoX-20B: An Open-Source Autoregressive Language Model](https://arxiv.org/abs/2204.06745)." In *Proceedings of the ACL Workshop on Challenges \& Perspectives in Creating Large Language Models*. 2022.

The following publications by other research groups use this library:
- Chi, Fan, Ramadge, and Rudnicky. "[KERPLE: Kernelized Relative Positional Embedding for Length Extrapolation](https://arxiv.org/abs/2205.09921)". _arXiv preprint arXiv:2205.09921_. 2022.
- Horawalavithana, Ayton, Sharma, Howland, Subramanian, Vasquez, Cosbey, Glenski, and Volkova. "[Foundation Models of Scientific Knowledge for Chemistry: Opportunities, Challenges and Lessons Learned](https://openreview.net/pdf?id=SLX-I2MHUZ9)." In *Proceedings of the ACL Workshop on Challenges \& Perspectives in Creating Large Language Models*. 2022.
- Kolak, Martins, Le Goues, and Hellendoorn. "[Patch Generation with Language Models: Feasibility and Scaling Behavior](https://openreview.net/forum?id=rHlzJh_b1-5)"." In *Proceedings of the Deep Learning for Code Workshop at ICLR*. 2022.
- Muennighoff, Niklas. "[SGPT: GPT Sentence Embeddings for Semantic Search](https://arxiv.org/abs/2202.08904)." *arXiv preprint arXiv:2202.08904*. 2022.
- Xu, Alon, Neubig, and Hellendoorn. "[A Systematic Evaluation of Large Language Models of Code](https://arxiv.org/abs/2202.13169)." In *Proceedings of the ICLR Workshop on Deep Learning For Code*. 2022.

## Acknowledgements

We run our experiments on a Kubernetes cluster generously provided by [CoreWeave](https://coreweave.com/) and a SLURM cluster provided by [Stability AI](https://stability.ai).
