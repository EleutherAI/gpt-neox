#!/usr/bin/env bash

#  --- USAGE ---
# $ kill_k8s.sh [name_suffix=$USER]

SUFFIX=${1:-$(whoami)}
DEPLOYMENT_NM='megatron-'"$SUFFIX"
MOUNT_NAME="$DEPLOYMENT_NM-ssd-cluster"

kubectl delete deploy/$DEPLOYMENT_NM
kubectl delete persistentvolumeclaims/$MOUNT_NAME
