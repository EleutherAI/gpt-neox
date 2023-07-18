#!/bin/bash
#BSUB -nnodes 92
#BSUB -W 2:00 
#BSUB -q batch
#BSUB -o gpt_neox_out.%J
#BSUB -e gpt_neox_err.%J
#BSUB -J gpt_neox
#BSUB -alloc_flags gpudefault
#BSUB -P CSC499


source /gpfs/alpine/csc499/scratch/$(whoami)/setup.sh 

#conda init bash

source /gpfs/alpine/csc499/scratch/kublaikhan1/miniconda3/etc/profile.d/conda.sh
conda activate robin

export TORCH_EXTENSIONS_DIR=/gpfs/alpine/csc499/scratch/kublaikhan1/latest_install/cache
# export this under /gpfs

TRAIN_PATH=/ccs/home/kublaikhan1/multimodal

# pls change this to your location of our repo

cd $TRAIN_PATH

bash /gpfs/alpine/csc499/scratch/$(whoami)/write_hostfile.sh
# pls find this shell script at GPT-NeoX on Summit
export DLTS_HOSTFILE=/gpfs/alpine/csc499/scratch/$(whoami)/hostfiles/$LSB_JOBID-hosts
export MASTER_ADDR=$(cat $LSB_DJOB_HOSTFILE | sort | uniq | grep -v batch | grep -v login | head -1)

python3 $TRAIN_PATH/deepy.py $TRAIN_PATH/train.py \
	--conf_dir $TRAIN_PATH/configs magma_pythia_410M.yml magma_setup.yml 


