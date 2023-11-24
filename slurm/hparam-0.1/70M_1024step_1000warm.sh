#!/bin/bash
#SBATCH --job-name="eleutherscaling"
#SBATCH --array=2
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
#SBATCH --output=70M_1024step_1000warm_%a_%A.out
#SBATCH --error=70M_1024step_1000warm_%a_%A.out
#SBATCH --time=3-00:00:00

# BYU cluster

# parameters, steps eval interval
declare -a args=(
    "0.001,0.0001,128" "0.01,0.001,128" "0.05,0.005,128" "0.1,0.01,128" "0.5,0.05,128," 
)
export SAVE_BASE_DIR="/home/za2514/compute/scaling/saved-weights/hparam-0.1"

export tuple="${args[$SLURM_ARRAY_TASK_ID]}"

# Unpack the tuple into named variables
IFS=',' read -ra tuple_array <<< "$tuple"
parameters="70M"
num_steps="1024"
warmup_iters="1000"
lr="${tuple_array[0]}"
min_lr="${tuple_array[1]}"
eval_interval="${tuple_array[2]}"

echo "3D job array parameters:" $parameters $num_steps $lr $min_lr $warmup_iters $eval_interval

run_name=${parameters}_1024step_${warmup_iters}warmup_${lr}lr

source /home/hailey81/miniconda3/bin/activate llmath_flashv2_fixed-ds

which python

export LD_LIBRARY_PATH=/home/hailey81/miniconda3/envs/llmath_flashv2_fixed-ds/lib/
export PATH=/home/hailey81/cuda_install/bin:$PATH

ln -s /home/hailey81/miniconda3/envs/llmath_flashv2_fixed-ds/bin/gcc/ ~/.local/bin/gcc
export PATH=$HOME/.local/bin:$PATH

export WANDB_MODE=offline

export TRAIN_DIR=/home/za2514/compute/scaling/gpt-neox

export LOG_BASE_DIR=${TRAIN_DIR}/logs

export CACHE=$TRAIN_DIR/.cache
export TRANSFORMERS_CACHE=$CACHE
export HF_DATASETS_CACHE=$CACHE
export HUGGINGFACE_HUB_CACHE=$CACHE

cd $TRAIN_DIR
pwd

python ./deepy.py train.py \
    --conf_dir ${TRAIN_DIR}/configs/hparam-0.1 base.yml ${parameters}_no-lr.yml \
    --train_iters $num_steps \
    --lr $lr \
    --min_lr $min_lr \
    --warmup_iters $warmup_iters \
    --lr_decay_iters $num_steps \
    --eval_interval $eval_interval \
    --checkpoint_factor $num_steps \
    --save ${SAVE_BASE_DIR}/${run_name} \
    --log_dir ${LOG_BASE_DIR}/${run_name} \
    --wandb_group $run_name
