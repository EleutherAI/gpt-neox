#!/bin/bash
#SBATCH --job-name="llmath_convert"
# #SBATCH --account=dw87
#SBATCH --comment="eleutherai"
#SBATCH --qos=dw87
#SBATCH --partition=dw
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8GB
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --open-mode=append
#SBATCH --output=llama_convert_correct-scheduler_step10000_%x_%j.out
#SBATCH --error=llama_convert_correct-scheduler_step10000_%x_%j.out
#SBATCH --time=2:00:00

# BYU cluster

source /home/hailey81/miniconda3/bin/activate llmath

which python

export LD_LIBRARY_PATH=/home/hailey81/miniconda3/envs/llmath/lib/
export PATH=/home/hailey81/cuda_install/bin:$PATH

ln -s /home/hailey81/miniconda3/envs/llmath/bin/gcc/ ~/.local/bin/gcc
export PATH=$HOME/.local/bin:$PATH

# export WANDB_API_KEY=07cebf97416da7fa921b74774ef771f52d4e49e9
# wandb login
export WANDB_MODE=offline

export TRAIN_DIR=/home/za2514/math-lm/gpt-neox/

cd ${TRAIN_DIR}
pwd

export PYTHONPATH=$TRAIN_DIR

python tools/checkpoints/convert_v1.0_to_hf.py --input_dir /home/za2514/saved-weights/proofgpt-v0.5/proofgpt-v0.5-llama-7b-correct_scheduler/global_step10000 --config_file /home/za2514/math-lm/gpt-neox/configs/v0.5/llama_7B_correct-scheduler.yml --output_dir /home/za2514/saved-weights/proofgpt-v0.5/proofgpt-v0.5-llama-7b-correct_scheduler_step10000-hf/

