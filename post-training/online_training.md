# Online Training

## Prerequisites
Want to use [REINFORCE](https://arxiv.org/abs/2402.14740) to train your model? First you'll need to build a custom vllm package.

[synth-vllm](https://github.com/SynthLabsAI/synth-vllm) is a fork of [vllm](https://github.com/vllm-project/vllm) maintained by [SynthLabs](https://www.synthlabs.ai/)
that has been modified to support using the weights in GPT-NeoX by sharing the GPU memory location of the model weights.

It currently supports Llama and Pythia models.

### Building the package

Here is a reference on how the package has been built before, using conda:
(Note this should be taken as a reference, and may not work as is due to your system configuration)

```bash
# cd to the synth vllm directory...
conda create -n vllm python=3.10
conda deactivate
conda activate vllm
conda install -y pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y nvidia/label/cuda-12.1.0::cuda-toolkit
conda install -y nvidia/label/cuda-12.1.0::cuda-cudart
conda install -y nvidia/label/cuda-12.1.0::cuda-compiler
conda install -y nvidia/label/cuda-12.1.0::cuda-nvcc
conda install -y nvidia/label/cuda-12.1.0::cuda-profiler-api
conda install -y nvidia/label/cuda-12.1.0::cuda-cudarty
conda install -y -c nvidia cuda-nvprof=12.1
conda install -y conda-forge::cuda-version=12.1
conda install -y gcc_linux-64=12.3.0
conda install -y -c conda-forge gxx_linux-64=12.3.0
pip install -e .
```

## Training

If you haven't already, run this command to generate a copy of the Llama-3 weights in GPT-NeoX format:
```bash
python tools/ckpts/convert_hf_llama_to_neox.py --tp 4 --model meta-llama/Meta-Llama-3-8B-Instruct --model_path checkpoints/neox_converted/llama3-8b-instruct
```

[online_example.sh](online_example.sh), [online_data_example_llama3.py](online_data_example_llama3.py) is an example of
how to train a model using the synth-vllm package on a single node.

This assumes you are using a conda environment with GPT-NeoX installed under the name `neox`.

To run the example, execute the following commands:

```bash
# It may be preferable to run these in two separate terminals
python post-training/online_data_example_llama3.py &
bash post-training/online_example.sh
```

This will train a model using the synth-vllm package on the llama3-8b-instruct model. It will optimize a positive reward
from a sentiment classifier.
