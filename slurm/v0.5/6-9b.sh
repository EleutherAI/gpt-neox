#!/bin/bash
#SBATCH --job-name="llmath7B"
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
#SBATCH --output=6-9b_%x_%j.out
#SBATCH --error=6-9b_%x_%j.out
#SBATCH --time=24:00:00

# BYU Cluster

source /home/hailey81/miniconda3/bin/activate llmath

which python

export LD_LIBRARY_PATH=/home/hailey81/miniconda3/envs/llmath/lib/
export PATH=/home/hailey81/cuda_install/bin:$PATH

# export WANDB_API_KEY=07cebf97416da7fa921b74774ef771f52d4e49e9
# wandb login

export TRAIN_DIR=/home/za2514/math-lm/gpt-neox/

cd $TRAIN_DIR
pwd

python ./deepy.py train.py --conf_dir /home/za2514/math-lm/gpt-neox/configs/v0.5/ 6-9B.yml
