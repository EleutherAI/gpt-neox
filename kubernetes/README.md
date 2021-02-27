## EleutherAI cluster

This directory contains code designed to facilitate working on our Kubernetes cluster. If you are an EleutherAI member and wish to be given access to the cluster, reach out to Stella or Sid.

If you are not an EleutherAI member, do not use any of the code in this directory.

### Using a cluster

If you already have a EleutherAI cluster setup for you, test to see if the cluster is working:
    
1. Copy data from cold storage to the shared mount: `cp -R /mnt/ssd-0/megatron-3d/data /mnt/ssd-cluster/data`
2. Example run `bash examples/ds_pretrain_gpt2_medium_pipe.sh`
    
## Cluster features

Setup:

* Use the "main node" as the entry point. This is the node with index 0
* All nodes have read/write access to a shared mount at `/mnt/ssd-cluster`. The default location for data for GPT-NEOX is set to `/mnt/ssd-cluster/data`
* All nodes have read access to a cold storage mount. This is where preprocessed data is kept `/mnt/ssd-0`
* A copy of the gpt-neox repo is cloned to `~/gpt-neox`

Tools (cd to `~/gpt-neox`):

* To kill a run: `bash tools/killall.sh`
* To copy a file to all nodes `bash tools/sync.sh $FILE`
* To copy a directory to all nodes `bash tools/syncdir.sh $DIR`
* To run a `git pull` command on all nodes `pdsh -w ^/job/hosts 'cd gpt-neox; git pull'`
* `/job/hostfile` and `/job/hosts` store the list of cluster nodes in Deepspeed and PDSH format respectively

CLI utils:
* `htop`: process monitor, CPU and memory utilisation
* `gpustat`: GPU utilisation
* `tmux`: use this so that when you disconnect you don't kill your run

## Setting up a cluster

Requires necessary permissions. To set-up a cluster for yourself:

1. `bash kubernetes/deploy_cluster.sh main 2` to deploy a 2 node cluster and clone the main branch of this repo to each node

To set-up a cluster for someone else (named `NAME`) without cluster permissions:

1. `bash kubernetes/deploy_cluster.sh main 2 NAME`
2. `bash kubernetes/public_cluster.sh NAME`

## Cluster management tools

* To write data to cold storage. Start the cold storage writer node: `bash kubernetes/deploy_data_writer.sh`
* To open a node: `bash kubernetes/open_pod.sh $NODENAME`
* To kill your cluster: `bash kubernetes/kill_k8s.sh`
