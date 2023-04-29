#!/bin/bash
#BSUB -nnodes 4
#BSUB -W 2:00 
#BSUB -q batch
#BSUB -o gpt_neox_out.%J
#BSUB -e gpt_neox_err.%J
#BSUB -J gpt_neox
#BSUB -alloc_flags gpudefault
#BSUB -P CSC499

source /gpfs/alpine/csc499/scratch/lfsm/setup.sh 

export TORCH_EXTENSIONS_DIR=/gpfs/alpine/scratch/lfsm/csc499/mycache/torch_extensions/

TRAIN_PATH=~/code/gpt-neox
cd $TRAIN_PATH

bash /gpfs/alpine/csc499/scratch/lfsm/write_hostfile.sh
export DLTS_HOSTFILE=/gpfs/alpine/csc499/scratch/lfsm/hostfiles/$LSB_JOBID-hosts

python $TRAIN_PATH/deepy.py $TRAIN_PATH/train.py \
	--conf_dir $TRAIN_PATH/configs summit-1-3B.yml summit_setup.yml 
