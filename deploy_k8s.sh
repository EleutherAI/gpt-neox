#!/usr/bin/env bash

BRANCH=${1:-main}

echo STARTING KUBERNETES USING CODE ON BRANCH $BRANCH. LOCAL CHANGES WILL NOT BE AVAILABLE

kubectl delete deploy/eleuther-neox
kubectl apply -f kubernetes/deploy_k8s.yml
ssh-keygen -t rsa -f id_rsa -N ""

echo Waiting for deploy to complete...
kubectl wait --for=condition=available --timeout=600s deployment/eleuther-neox || { echo 'Deployment failed' ; exit 1; }

echo Generate hosts file
kubectl get pods -o wide | grep eleuther-neox | awk '{print $6 " slots=8"}' > hostfile
export MAIN_ID=$(kubectl get pods | grep eleuther-neox | awk '{print $1}' | head -n 1)

echo Copying ssh keys to main node:
echo $MAIN_ID
kubectl cp $PWD/hostfile $MAIN_ID:/job
kubectl cp $PWD/id_rsa $MAIN_ID:/root/.ssh

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
    kubectl cp $PWD/authorized_keys $id:/root/.ssh/
    echo $pod_cmd | kubectl exec --stdin $id -- /bin/bash
done
rm authorized_keys hostfile
rm id_rsa*

echo Remote shell into main $MAIN_ID
kubectl exec --stdin --tty $MAIN_ID -- /bin/bash
