# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

GPT-NeoX is EleutherAI's library for training large-scale autoregressive language models on GPUs. Built on NVIDIA's Megatron-LM and enhanced with DeepSpeed, it supports distributed training with advanced parallelism techniques and cutting-edge architectures.

## Common Development Commands

### Environment Setup
```bash
# Install core dependencies
pip install -r requirements/requirements.txt

# Optional monitoring tools
pip install -r requirements/requirements-wandb.txt
pip install -r requirements/requirements-tensorboard.txt

# For flash attention support
pip install -r requirements/requirements-flashattention.txt

# Install pre-commit hooks (required for contributing)
pre-commit install
conda install clang-format
```

### Data Preparation
```bash
# Download and tokenize default datasets
python prepare_data.py -d ./data

# Tokenize with specific tokenizer
python prepare_data.py -d ./data -t HFTokenizer --vocab-file ./20B_checkpoints/20B_tokenizer.json pile_subset

# Preprocess custom JSONL data
python tools/datasets/preprocess_data.py \
    --input ./data/mydataset.jsonl.zst \
    --output-prefix ./data/mydataset \
    --vocab ./data/gpt2-vocab.json \
    --merge-file gpt2-merges.txt \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --append-eod
```

### Training
```bash
# Train with single config
python deepy.py train.py configs/125M.yml

# Train with merged configs
python deepy.py train.py -d configs 125M.yml local_setup.yml

# Multi-node training
python deepy.py train.py configs/20B.yml configs/cluster_setup.yml
```

### Evaluation
```bash
# Evaluate on specific tasks
python deepy.py eval.py -d configs your_configs.yml --eval_tasks lambada hellaswag piqa sciq

# Multiple evaluation tasks
python deepy.py eval.py configs/125M.yml configs/local_setup.yml --eval_tasks task1 task2
```

### Text Generation
```bash
# Generate text (unconditional/conditional)
python deepy.py generate.py -d configs 125M.yml local_setup.yml text_generation.yml

# Interactive generation
python deepy.py generate.py configs/model.yml --text_gen_type interactive
```

### Testing
```bash
# Run all tests with forking (required for CUDA tests)
pytest --forked

# Run unit tests only
pytest tests/unit/

# Run specific test file
pytest tests/model/test_fused_kernels.py

# CPU-only tests
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python pytest tests -m cpu

# Model convergence tests
cd tests/model/ && pytest run_sanity_check.py
```

### Linting and Formatting
```bash
# Run pre-commit hooks manually
pre-commit run --all-files

# Black formatting is applied automatically via pre-commit
```

### Checkpoint Management
```bash
# Convert NeoX to HuggingFace format
python tools/ckpts/convert_neox_to_hf.py \
    --input_dir /path/to/model/global_stepXXX \
    --config_file your_config.yml \
    --output_dir hf_model/save/location \
    --precision {auto,fp16,bf16,fp32} \
    --architecture {neox,mistral,llama}

# Convert HF to NeoX format
python tools/ckpts/convert_hf_to_sequential.py

# Convert raw LLaMA weights to NeoX
python tools/ckpts/convert_raw_llama_weights_to_neox.py \
    --input_dir /path/to/model/7B \
    --model_size 7B \
    --output_dir /path/to/save/ckpt \
    --num_output_shards <TENSOR_PARALLEL_SIZE>
```

## Architecture Overview

### Entry Points
- `deepy.py` - Main launcher that wraps DeepSpeed functionality
- `train.py` - Training and finetuning models
- `eval.py` - Model evaluation on downstream tasks
- `generate.py` - Text generation (interactive, conditional, unconditional)
- `prepare_data.py` - Data preprocessing and tokenization

### Configuration System
The project uses YAML-based configuration with config merging support:
- Model configs: `configs/125M.yml`, `configs/6-7B.yml`, `configs/20B.yml`, etc.
- Architecture configs: `configs/llama/`, `configs/mistral/`, `configs/mamba/`
- Training configs can be merged at runtime for flexibility
- All hyperparameters controlled via YAML files

### Parallelism Architecture
Supports multiple parallelism strategies:
- Data Parallel (DP)
- Tensor/Model Parallel (TP/MP)
- Pipeline Parallel (PP)
- ZeRO optimization (stages 1-3)
- 3D parallelism combining all above

### Key Dependencies
- **DeeperSpeed**: EleutherAI's DeepSpeed fork (custom installation required)
- **Core**: PyTorch, transformers, tokenizers, pybind11
- **Optional**: Flash Attention, Transformer Engine, apex
- **Monitoring**: Weights & Biases, TensorBoard, Comet

### Project Structure
- `megatron/` - Core model implementations and training logic
- `configs/` - YAML configuration files for models and training
- `tools/` - Utilities for data processing, checkpoints, and cluster management
- `tests/` - Unit and model convergence tests
- `requirements/` - Dependency specifications for different features

### Development Workflow

1. **Configuration**: Create or modify YAML configs for your model/training setup
2. **Data Pipeline**: Prepare data using provided scripts (supports JSONL format with "text" field)
3. **Training**: Launch training with appropriate parallelism settings
4. **Monitoring**: Use WandB/TensorBoard for tracking metrics
5. **Evaluation**: Run evaluation harness on checkpoints
6. **Export**: Convert checkpoints to HuggingFace format for deployment

### Multi-Node Training
For distributed training across nodes:
1. Create hostfile with node IPs and slots
2. Configure launcher (pdsh, MPI, or SLURM)
3. Set parallelism dimensions in config
4. Use appropriate launch command based on cluster setup

### Testing Guidelines
- Always run tests with `--forked` flag for CUDA functionality
- Unit tests focus on individual components
- Model tests verify convergence behavior
- Pre-commit hooks enforce code formatting standards