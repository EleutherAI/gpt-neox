#!/usr/bin/env bash

#  --- USAGE ---
# $ deploy_k8.sh [branch=main] [n_nodes=4] [name_suffix=$USER] [image]
# You need to install yq

# Check yq
yq &> /dev/null || { echo 'You need to install `yq >= v4`. `brew install yq` or `pip install yq`' ; exit 1; }

WD_BRANCH=$(git branch  --no-color --show-current)
WD_BRANCH="${WD_BRANCH/\//-}"  # remove forward slashes and replace with underscore
DEFAULT_IMAGE="leogao2/megatron-3d:sha-b8f5852"

BRANCH=${1:-main}
N_NODES=${2:-4}
SUFFIX=${3:-$(whoami)}
IMAGE=${4:-$DEFAULT_IMAGE}

DEPLOYMENT_NM='megatron-'"$SUFFIX"
WD=`dirname "$BASH_SOURCE"`

echo BRANCH $BRANCH. N-NODES $N_NODES. DEPLOYMENT NAME $DEPLOYMENT_NM. DOCKER IMAGE $IMAGE.

# Obtain wandb API key
WANDB_APIKEY=$(python $WD/get_wandb_api_key.py)
if [ -n "$WANDB_APIKEY" ]
then
      echo "wandb.ai API successfully obtained"
fi

# Generate ssh key pair and post start script
echo Generate SSH key pair
rm $WD/id_rsa*
ssh-keygen -t rsa -f $WD/id_rsa -N "" 

post_start_script="
echo 'export DATA_DIR=/mnt/ssd-cluster/data' >> /home/mchorse/.bashrc;
sudo cp /secrets/id_rsa.pub /home/mchorse/.ssh/authorized_keys;
sudo chown mchorse:mchorse /home/mchorse/.ssh/authorized_keys;
sudo chown -R mchorse:mchorse /home/mchorse/.ssh;
chmod 600 /home/mchorse/.ssh/authorized_keys;
chmod 700 /home/mchorse/.ssh;
cd /home/mchorse;
git clone --branch $BRANCH https://github.com/EleutherAI/megatron-3d.git;
apt-get update -y;
apt-get install -y libpython3-dev;
sudo mkdir /job;
sudo chown mchorse:mchorse /job;
"
if [ -n "$WANDB_APIKEY" ]
then
      post_start_script+=" wandb login $WANDB_APIKEY; "
fi

echo $post_start_script > $WD/post_start_script.sh

# Add ssh key to k8 secrets and post start script
DATE=$(date +%s)
SECRET_NM="$DEPLOYMENT_NM-$DATE"
kubectl create secret generic $SECRET_NM \
  --from-file=id_rsa.pub=$WD/id_rsa.pub \
  --from-file=post_start_script.sh=$WD/post_start_script.sh

# Template k8 configuration - deployment
MOUNT_NAME="$DEPLOYMENT_NM-ssd-cluster"
cat $WD/k8s_spec.yml |
yq e '.metadata.name = "'"$DEPLOYMENT_NM"\" - |
yq e '.spec.replicas = '"$N_NODES" - |
yq e '.spec.template.spec.volumes[1].secret.secretName = "'"$SECRET_NM"\" - |
yq e '.spec.template.spec.containers[0].image = "'"$IMAGE"\" - |
yq e '.spec.template.spec.volumes[3].persistentVolumeClaim.claimName = "'"$MOUNT_NAME"\" - > $WD/k8s_spec_temp.yml

# Template k8 configuration - shared mount
cat $WD/k8_spec_ssd-cluster.yml |
yq e '.metadata.name = "'"$MOUNT_NAME"\" - > $WD/k8_spec_ssd-cluster_temp.yml

# Delete previous and setup deployment
kubectl delete deploy/$DEPLOYMENT_NM || { echo 'No previous deployment'; }
kubectl delete persistentvolumeclaims/$MOUNT_NAME || { echo 'No previous mount'; }

kubectl apply -f $WD/k8_spec_ssd-cluster_temp.yml
kubectl apply -f $WD/k8s_spec_temp.yml

echo Waiting for deploy to complete...
kubectl wait --for=condition=available --timeout=600s deployment/$DEPLOYMENT_NM || { echo 'Deployment failed' ; exit 1; }

echo Generate hosts file
kubectl get pods -o wide | grep $DEPLOYMENT_NM | awk '{print $6 " slots=8"}' > $WD/hostfile
cat $WD/hostfile | cut -f1 -d' ' > $WD/hosts
export MAIN_ID=$(kubectl get pods | grep $DEPLOYMENT_NM | awk '{print $1}' | head -n 1)

echo Copying ssh key and host file to main node:
echo $MAIN_ID
kubectl cp $WD/hostfile $MAIN_ID:/job
kubectl cp $WD/hosts $MAIN_ID:/job
kubectl cp $WD/id_rsa $MAIN_ID:/home/mchorse/.ssh

rm $WD/id_rsa* $WD/hostfile $WD/hosts $WD/k8s_spec_temp.yml $WD/k8_spec_ssd-cluster_temp.yml $WD/post_start_script.sh

echo Remote shell into main $MAIN_ID
kubectl exec --stdin --tty $MAIN_ID -- /bin/bash
