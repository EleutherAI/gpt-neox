#!/bin/bash
set -ex
# This path refers to hostfile path
# all nodes in the hostfile will be used

CONFIG_FILE=${1:-"configs/sai_15B.yml"}
CONTAINER_NAME=${2:-"sliuxl__stable_ckpt_v00"}
CONTAINER_NAME=${2:-"yongyanr-test__leecheng-pt1_13_sai_nsys"}
DIR=${3:-"/fsx/sliuxl/EleutherAI/gpt-neox"}
HOSTFILE=${4:-"$DIR/hostfile"}

MPI_HOSTFILE=${HOSTFILE}_mpi

cp ${HOSTFILE} ${MPI_HOSTFILE}
sed -i 's/$/ slots=8/g' ${MPI_HOSTFILE}
cat ${MPI_HOSTFILE}

NUM_NODES=`cat $HOSTFILE | wc -l`
# kill any prev processes left
SLURM_HOSTFILE="$HOSTFILE srun -N $NUM_NODES sudo pkill -9 python"
JID=$RANDOM


sleep 3


RUN_CMD="cat ${MPI_HOSTFILE} && \
        cd $DIR && \
        /opt/amazon/openmpi/bin/mpirun -N 8 --hostfile ${MPI_HOSTFILE} \
        --mca pml ^cm --mca plm_rsh_no_tree_spawn 1 --mca btl_tcp_if_exclude lo,docker0 \
        --mca plm_rsh_num_concurrent `cat $HOSTFILE | wc -l`\
        --allow-run-as-root --mca btl_vader_single_copy_mechanism none --oversubscribe --tag-output \
        -x FI_PROVIDER=efa -x RDMAV_FORK_SAFE=1 -x FI_EFA_USE_DEVICE_RDMA=1 \
        -x NCCL_SOCKET_IFNAME="^lo,docker0" -x NCCL_ALGO=ring -x NCCL_PROTO=simple \
        -x LD_LIBRARY_PATH -x PATH \
        /opt/conda/bin/python train_mpi.py --conf $CONFIG_FILE"

# ------
# Mark all nodes in the cluster as busy so slurm doesn't think they are idle
# When this script returns, this ghost job will exit and nodes will be freed due to the trap command
SLURM_HOSTFILE=${HOSTFILE} srun --job-name=hb_"$JID" --quit-on-interrupt --quiet sleep infinity >/dev/null 2>&1 &
trap "scancel -n hb_${JID}; scancel -n train_${JID}" EXIT
# runs the mpirun command generated above on 1st node in the hostfile
SLURM_HOSTFILE=${HOSTFILE} srun --job-name=train_"$JID" -N 1 docker exec ${CONTAINER_NAME} bash -c "${RUN_CMD}"
rm ${MPI_HOSTFILE}
