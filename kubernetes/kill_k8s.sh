#!/usr/bin/env bash

#  --- USAGE ---
# $ kill_k8s.sh [name_suffix=$USER]

SUFFIX=${1:-$(whoami)}
CLUSTER_NM='neox-'"$SUFFIX"
MOUNT_NAME="$CLUSTER_NM-ssd-cluster"
SERVICE_NM="$CLUSTER_NM-service"

kubectl delete statefulsets/$CLUSTER_NM
kubectl delete persistentvolumeclaims/$MOUNT_NAME
kubectl delete services/$SERVICE_NM
