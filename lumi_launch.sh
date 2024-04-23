#!/bin/bash

export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export OMP_NUM_THREADS=1

echo "Run on $SLURMD_NODENAME ($SLURM_PROCID/$WORLD_SIZE)," \
     "master $MASTER_ADDR port $MASTER_PORT," \
     "GPUs $SLURM_GPUS_ON_NODE"

CMD="
source /opt/miniconda3/bin/activate pytorch
python3 $@
"

REAL_PWD="$(realpath "$PWD")"

singularity exec --pwd "$REAL_PWD" "$CONTAINER" bash -c "$CMD"
