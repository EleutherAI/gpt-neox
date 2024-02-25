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
#SBATCH --output=muinstruct16x_%a_%A.out
#SBATCH --error=muinstruct16x_%a_%A.out
#SBATCH --time=3-00:00:00

# BYU cluster

declare -a lrs=("1e-5")

lr="${lrs[$SLURM_ARRAY_TASK_ID]}"

run_name="llemma_7b_muinstruct16x_10epoch_${lr}lr_v0.2"

source /home/hailey81/miniconda3/bin/activate llmath_flashv2_fixed-ds

which python

export LD_LIBRARY_PATH=/home/hailey81/miniconda3/envs/llmath_flashv2_fixed-ds/lib/
export PATH=/home/hailey81/cuda_install/bin:$PATH

ln -s /home/hailey81/miniconda3/envs/llmath_flashv2_fixed-ds/bin/gcc/ ~/.local/bin/gcc
export PATH=$HOME/.local/bin:$PATH

# export WANDB_API_KEY=07cebf97416da7fa921b74774ef771f52d4e49e9
# wandb login
export WANDB_MODE=offline

export TRAIN_DIR=/home/za2514/compute/Llemma/gpt-neox_math-lm-2-rotary/

cd $TRAIN_DIR
pwd

SAVE_DIR="/nobackup/scratch/usr/za2514/saved-weights/llemma-instruct/${run_name}"
CONF_DIR=/home/za2514/compute/Llemma/gpt-neox_math-lm-2-rotary/configs/llemma-instruct
CONF_FILE=muinstruct16x.yml
CONF_PATH=${CONF_DIR}/${CONF_FILE}

python ./deepy.py train.py \
    --conf_dir $CONF_DIR $CONF_FILE \
    --save $SAVE_DIR \
    --wandb_group ${run_name} \
    --log_dir /home/za2514/logs/${run_name} \
    --lr ${lr}

export PYTHONPATH=$TRAIN_DIR

HF_DIR="${SAVE_DIR}-hf"
python tools/convert_llama_sequential_to_hf.py --input_dir ${SAVE_DIR}/global_step1599 --config_file ${CONF_PATH} --output_dir $HF_DIR && cp -v /nobackup/scratch/usr/za2514/codellama/tokenizer.model $HF_DIR

wait
