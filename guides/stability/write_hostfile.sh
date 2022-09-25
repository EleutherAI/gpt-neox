#!/bin/bash
hostfile=/fsx/zphang/hostfiles/hosts_$SLURM_JOBID

for i in `scontrol show hostnames $SLURM_NODELIST`
do
    echo $i slots=8 >>$hostfile
done