#!/bin/bash
#SBATCH --job-name="tokenization"
# #SBATCH --account=dw87
#SBATCH --comment="eleutherai"
#SBATCH --qos=dw87
#SBATCH --partition=dw
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8GB
#SBATCH --open-mode=append
#SBATCH --output=tokenize_open-web-math-decontaminated_train_%j.out
#SBATCH --error=tokenize_open-web-math-decontaminated_train_%j.out
#SBATCH --time=1-00:00:00

# BYU cluster
SUBSET=open-web-math
SPLIT=train

DATA_DIR="/home/za2514/compute/open-web-math-decontaminated_llama/${SPLIT}"
TOKENIZER_TYPE=SPMTokenizer
VOCAB_FILE=/home/za2514/downloaded-weights/llama/tokenizer.model
LOG_FILE=${DATA_DIR}/${SUBSET}/tokenization.log

source /home/hailey81/miniconda3/bin/activate llmath

which python

NEOX_DIR=/home/za2514/compute/math-lm/gpt-neox
cd $NEOX_DIR
pwd

export PYTHONPATH="${PYTHONPATH}:${NEOX_DIR}"

python prepare_data.py $SUBSET -d $DATA_DIR -t $TOKENIZER_TYPE --vocab-file $VOCAB_FILE | tee $LOG_FILE
echo "exited successfully"
