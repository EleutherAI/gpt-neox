#!/bin/bash
source /opt/miniconda3/bin/activate pytorch
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

echo "Run on $SLURMD_NODENAME ($SLURM_PROCID/$WORLD_SIZE)," \
     "master $MASTER_ADDR port $MASTER_PORT," \
     "GPUs $SLURM_GPUS_ON_NODE"


cd /projappl/project_462000558/villekom/gpt-neox
python3 $@


#REAL_PWD="$(realpath "$PWD")"
#singularity exec $CONTAINER ds_report
#"$CMD"
