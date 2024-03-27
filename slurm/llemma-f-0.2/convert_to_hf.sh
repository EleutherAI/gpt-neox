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
#SBATCH --output=convert_to_hf_%j.out
#SBATCH --error=convert_to_hf_%j.out
#SBATCH --time=1:00:00

# BYU cluster


OUT_NAME="llemma_f_v0.2_no-terms_1e-5lr"
echo $OUT_NAME
INPUT_DIR="/home/za2514/saved-weights/llemma-f-0.2/llemma_7b_f_v0.2_main_no-terms_1e-5lr/global_step9500"
OUT_DIR="/home/za2514/saved-weights/llemma-f-0.2/${OUT_NAME}/"
CONFIG_FILE="/home/za2514/Llemma/gpt-neox-math-lm-2-rotary/configs/llemma-f-0.2/main_no-terms.yml"

source /home/hailey81/miniconda3/bin/activate llmath

which python

export LD_LIBRARY_PATH=/home/hailey81/miniconda3/envs/llmath/lib/
export PATH=/home/hailey81/cuda_install/bin:$PATH

ln -s /home/hailey81/miniconda3/envs/llmath/bin/gcc/ ~/.local/bin/gcc
export PATH=$HOME/.local/bin:$PATH

export WANDB_MODE=offline

export TRAIN_DIR=/home/za2514/Llemma/gpt-neox-math-lm-2-rotary

cd ${TRAIN_DIR}
pwd

export PYTHONPATH=$TRAIN_DIR

python tools/convert_llama_sequential_to_hf.py --input_dir ${INPUT_DIR} --config_file ${CONFIG_FILE} --output_dir ${OUT_DIR}

cp -v /home/za2514/codellama/tokenizer.model $OUT_DIR

echo "exited successfully from ${string}"
