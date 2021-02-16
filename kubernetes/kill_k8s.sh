#!/usr/bin/env bash

#  --- USAGE ---
# $ kill_k8s.sh [name_suffix=$USER]

SUFFIX=${1:-$(whoami)}
CLUSTER_NM='neox-'"$SUFFIX"
MOUNT_NAME="$CLUSTER_NM-ssd-cluster"

kubectl delete statefulsets/$CLUSTER_NM
kubectl delete persistentvolumeclaims/$MOUNT_NAME
