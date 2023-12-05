#!/bin/bash

GPUS_PER_NODE=8
hostfile_folder=/admin/home-jinwooahn/repos/exp/hostfiles  # 본인 호스트파일 저장 폴더로 변경
mkdir -p $hostfile_folder

# need to add the current slurm jobid to hostfile name so that we don't add to previous hostfile
hostfile=$hostfile_folder/hosts_$SLURM_JOBID
echo "##################################################################################"
echo "HOSTFILE=${hostfile}"

# be extra sure we aren't appending to a previous hostfile
rm $hostfile &> /dev/null

# loop over the node names
for i in `scontrol show hostnames $SLURM_NODELIST`
do
        # add a line to the hostfile
        echo $i slots=$GPUS_PER_NODE >>$hostfile
done
echo "##################################################################################"
