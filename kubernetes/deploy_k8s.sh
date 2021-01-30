#!/usr/bin/env bash

#  --- USAGE ---
# ./deploy_k8.sh [branch=main] [n_nodes=4] [name_suffix=$USER]

BRANCH=${1:-main}
N_NODES=${2:-4}
SUFFIX=${3:-$(whoami)}

DEPLOYMENT_NM='eleuther-neox-'"$SUFFIX"
WD=`dirname "$BASH_SOURCE"`

echo BRANCH $BRANCH. N-NODES $N_NODES. DEPLYMENT NAME $DEPLOYMENT_NM.

# Template k8 configuration
yq e '.metadata.name = "'"$DEPLOYMENT_NM"\" $WD/k8s_spec.yml |
yq e '.spec.replicas = '"$N_NODES" - > $WD/k8s_spec_temp.yml


kubectl delete deploy/$DEPLOYMENT_NM
kubectl apply -f $WD/k8s_spec_temp.yml
ssh-keygen -t rsa -f id_rsa -N ""

echo Waiting for deploy to complete...
kubectl wait --for=condition=available --timeout=600s deployment/$DEPLOYMENT_NM || { echo 'Deployment failed' ; exit 1; }

echo Generate hosts file
kubectl get pods -o wide | grep eleuther-neox | awk '{print $6 " slots=8"}' > hostfile
export MAIN_ID=$(kubectl get pods | grep eleuther-neox | awk '{print $1}' | head -n 1)

echo Copying ssh keys to main node:
echo $MAIN_ID
kubectl cp $WD/hostfile $MAIN_ID:/job
kubectl cp $WD/id_rsa $MAIN_ID:/root/.ssh

mv id_rsa.pub authorized_keys

pod_cmd="
chmod 600 ~/.ssh/authorized_keys;
chmod 700 ~/.ssh;
chown -R root /root/.ssh;
rm -r *;
git clone --single-branch --branch $BRANCH https://github.com/EleutherAI/gpt-neox.git .;
pip uninstall -y deepspeed;
pip install git+git://github.com/EleutherAI/DeeperSpeed@main;
"

for id in $(kubectl get pods | grep eleuther-neox | awk '{print $1}')
do
    echo Copying keys and cloning repo to $id
    kubectl cp $WD/authorized_keys $id:/root/.ssh/
    kubectl cp $WD/authorized_keys $id:/root/.ssh/
    echo $pod_cmd | kubectl exec --stdin $id -- /bin/bash
done
rm authorized_keys hostfile
rm id_rsa*

echo Remote shell into main $MAIN_ID
kubectl exec --stdin --tty $MAIN_ID -- /bin/bash
