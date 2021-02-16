#!/usr/bin/env bash

# open_pod.sh [pod_name]

kubectl exec --stdin --tty $1 -- /bin/bash