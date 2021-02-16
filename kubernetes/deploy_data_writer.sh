#!/usr/bin/env bash

#  --- USAGE ---
# $ deploy_cluster.sh [branch] [image]
# You need to install yq

# Check yq
yq &> /dev/null || { echo 'You need to install `yq >= v4`. `brew install yq` or `pip install yq`' ; exit 1; }

DEFAULT_IMAGE="leogao2/gpt-neox:sha-ef55a42"

BRANCH=${1:-main}
IMAGE=${4:-$DEFAULT_IMAGE}

DEPLOYMENT_NM='neox-data-writer'
WD=`dirname "$BASH_SOURCE"`

echo DEPLOYMENT NAME $DEPLOYMENT_NM. BRANCH $BRANCH. DOCKER IMAGE $IMAGE.

post_start_script="
echo 'export DATA_DIR=/mnt/ssd-0/megatron-3d/data' >> /home/mchorse/.bashrc;
git clone --branch $BRANCH https://github.com/EleutherAI/gpt-neox.git;
"
echo $post_start_script > $WD/post_start_script_dw.sh

# Add ssh key to k8 secrets and post start script
DATE=$(date +%s)
SECRET_NM="$DEPLOYMENT_NM-$DATE"
kubectl create secret generic $SECRET_NM \
  --from-file=post_start_script.sh=$WD/post_start_script_dw.sh

# Template k8 configuration
cat $WD/k8s_spec_data_writer.yml |
yq e '.metadata.name = "'"$DEPLOYMENT_NM"\" - |
yq e '.spec.template.spec.volumes[1].secret.secretName = "'"$SECRET_NM"\" - |
yq e '.spec.template.spec.containers[0].image = "'"$IMAGE"\" - > $WD/k8s_spec_data_writer_temp.yml

# Delete previous and setup deployment
kubectl delete deploy/$DEPLOYMENT_NM || { echo 'No previous deployment'; }
kubectl apply -f $WD/k8s_spec_data_writer_temp.yml

echo Waiting for deploy to complete...
kubectl wait --for=condition=available --timeout=600s deployment/$DEPLOYMENT_NM || { echo 'Deployment failed' ; exit 1; }

export MAIN_ID=$(kubectl get pods | grep $DEPLOYMENT_NM | awk '{print $1}' | head -n 1)

rm $WD/k8s_spec_data_writer_temp.yml $WD/post_start_script_dw.sh

echo Remote shell into main $MAIN_ID
kubectl exec --stdin --tty $MAIN_ID -- /bin/bash
