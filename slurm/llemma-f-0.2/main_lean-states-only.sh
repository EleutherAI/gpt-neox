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
#SBATCH --array=0
#SBATCH --output=llemma-f-0.1-main-lean-states-only_%a_%A.out
#SBATCH --error=llemma-f-0.1-main-lean-states-only_%a_%A.out
#SBATCH --time=3-00:00:00
#SBATCH --exclude=dw-1-2,dw-2-1,dw-2-2

# BYU cluster

nvidia-smi

declare -a lrs=("1e-5" "3e-5" "9e-5")

lr="${lrs[$SLURM_ARRAY_TASK_ID]}"

run_name="main_lean-states-only_${lr}lr"

source /home/hailey81/miniconda3/bin/activate llmath_flashv2_fixed-ds

which python

export LD_LIBRARY_PATH=/home/hailey81/miniconda3/envs/llmath_flashv2_fixed-ds/lib/
export PATH=/home/hailey81/cuda_install/bin:$PATH

ln -s /home/hailey81/miniconda3/envs/llmath_flashv2_fixed-ds/bin/gcc/ ~/.local/bin/gcc
export PATH=$HOME/.local/bin:$PATH

# export WANDB_API_KEY=07cebf97416da7fa921b74774ef771f52d4e49e9
# wandb login
export WANDB_MODE=offline

export TRAIN_DIR=/home/za2514/Llemma/gpt-neox-math-lm-2-rotary/

cd $TRAIN_DIR
pwd

python ./deepy.py train.py \
    --conf_dir /home/za2514/Llemma/gpt-neox-math-lm-2-rotary/configs/llemma-f-0.2 main_lean-states-only.yml \
    --save "/home/za2514/saved-weights/llemma-f-0.2/llemma_7b_f_v0.2_${run_name}" \
    --wandb_group ${run_name} \
    --log_dir /home/za2514/logs/${run_name} \
    --lr ${lr}
