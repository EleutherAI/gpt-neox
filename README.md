# LLeMA Pretraining

This is a modified version of the `EleutherAI/GPT-NeoX` repository used for the Llemma project. This branch diverged from `main` at commit `009018e`. This branch implements the following features that are not present in `009018e` (some of these features may have subsequently been merged into `main`):
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- Grouped Query Attention
- A numerical precision fix for RoPE    
- Saving checkpoints to Amazon S3.

The remaining portion of this `README` contains instructions to replicate pretraining of the LLeMA models. 

Training was performed across 256 A100 GPUs. We include configuration files and sample SLURM job script for the library to replicate training on a SLURM-managed cluster.

We additionally provide public training logs via Weights and Biases, which can be found in a report [here](https://api.wandb.ai/links/eleutherai/7db5ehxl).

## Replicating Training


### Set up environment

We provide a file containing a dump of our training environment.

You can install all required packages via
```bash
pip install -r requirements.txt
```
Make sure you are installing https://github.com/EleutherAI/DeeperSpeed/tree/new-fix for your DeepSpeed version and install fused kernels for GPT-NeoX via `python ./megatron/fused_kernels/setup.py install` from within your GPT-NeoX install.


### Converting Llama 2 checkpoints into NeoX format

First, download CodeLlama 7b or 34b from the Meta AI repo and rename the download folder to 7B or 34B within the CodeLlama repository.

Then, to convert either model into the format expected by GPT-NeoX for checkpoints:

Sample command for 7b Meta->NeoX format:
```bash
python convert_raw_llama_weights_to_hf.py --input_dir /path/to/codellama/repo --config_file /path/to/this/repo/math-lm/pretraining/llemma_7b.yml --output_dir /path/to/save/into/ --num_output_shards {TP_DEGREE, we use 2}
```

Sample command for 34b Meta->NeoX format:
(Requires large amounts of GPU VRAM or CPU RAM. Pass `CUDA_VISIBLE_DEVICES=""` to perform conversion on CPU. 34b conversion may take a while)
```bash
CUDA_VISIBLE_DEVICES="" python convert_raw_llama_weights_to_hf.py --input_dir /path/to/codellama/repo --config_file /path/to/this/repo/math-lm/pretraining/llemma_34b.yml --output_dir /path/to/save/into/ --num_output_shards {TP_DEGREE, we use 8}
```


### Check Out Codebase

Next, check out the commit used to train the model you are replicating.

* 7b / 34b: https://github.com/EleutherAI/gpt-neox/commit/{this_commit_hash}

### Launching Training

Then, edit the provided YML files to set paths based on your own system's saved locations for checkpoints and data files, and edit the SLURM job script as specified (using ) or run the job across multiple nodes using your own system's orchestration.

**Tip**: Note that the global batch size will be scaled by your number of nodes. Therefore, if running on a number of nodes different from 32 you should scale gradient accumulation steps accordingly. 

We used a batch size of 4M tokens. To calculate global batch size, you should compute `seq_len * num_gpus * ( train_microbatch_size_per_gpu * gradient_accumulation_steps) / (model_parallel_size * max(pipeline_parallel_size, 1))` .


## Contents

The files in this folder are as follows:

* `34b_launch_script.sh` contains a skeleton SLURM job script to launch training with NeoX across 32 nodes.

* `configs/data_mixture.yml` contains a list of the domain weights for the final training run.

* `configs/llemma_7b.yml` is a cleaned-up version of the config file used to train Llemma-7b.

* `configs/llemma_34b.yml` is a cleaned-up version of the config file used to train Llemma-34b.

* `requirements.txt` is a dump of the virtual environmment used in training, created via `pip freeze`.
