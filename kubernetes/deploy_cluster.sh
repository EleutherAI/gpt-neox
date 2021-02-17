#!/usr/bin/env bash

#  --- USAGE ---
# $ deploy_k8.sh [branch=main] [n_nodes=4] [name_suffix=$USER] [image]
# You need to install yq

# Check yq
yq &> /dev/null || { echo 'You need to install `yq >= v4`. `brew install yq` or `pip install yq`' ; exit 1; }

WD_BRANCH=$(git branch  --no-color --show-current)
WD_BRANCH="${WD_BRANCH/\//-}"  # remove forward slashes and replace with underscore
DEFAULT_IMAGE="leogao2/gpt-neox:sha-ef55a42"

BRANCH=${1:-main}
N_NODES=${2:-4}
SUFFIX=${3:-$(whoami)}
IMAGE=${4:-$DEFAULT_IMAGE}

CLUSTER_NM='neox-'"$SUFFIX"
WD=`dirname "$BASH_SOURCE"`

echo BRANCH $BRANCH. N-NODES $N_NODES. CLUSTER NAME $CLUSTER_NM. DOCKER IMAGE $IMAGE.

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
echo 'export WANDB_TEAM=eleutherai' >> /home/mchorse/.bashrc;
echo 'export DS_EXE=/home/mchorse/gpt-neox/deepy.py' >> /home/mchorse/.bashrc;
sudo cp /secrets/id_rsa.pub /home/mchorse/.ssh/authorized_keys;
sudo chown mchorse:mchorse /home/mchorse/.ssh/authorized_keys;
sudo chown -R mchorse:mchorse /home/mchorse/.ssh;
chmod 600 /home/mchorse/.ssh/authorized_keys;
chmod 700 /home/mchorse/.ssh;
cd /home/mchorse;
git clone --branch $BRANCH https://github.com/EleutherAI/gpt-neox.git;
sudo apt-get update -y;
sudo apt-get install -y libpython3-dev;
sudo mkdir -p /job;
sudo chown mchorse:mchorse /job;
"
if [ -n "$WANDB_APIKEY" ]
then
      post_start_script+=" wandb login $WANDB_APIKEY; "
fi

echo $post_start_script > $WD/post_start_script.sh

# Add ssh key to k8 secrets and post start script
DATE=$(date +%s)
SECRET_NM="$CLUSTER_NM-$DATE"
kubectl create secret generic $SECRET_NM \
  --from-file=id_rsa.pub=$WD/id_rsa.pub \
  --from-file=post_start_script.sh=$WD/post_start_script.sh

# Template k8 configuration - deployment
MOUNT_NAME="$CLUSTER_NM-ssd-cluster"
cat $WD/k8s_spec.yml |
yq e '.metadata.name = "'"$CLUSTER_NM"\" - |
yq e '.spec.serviceName = "'"$CLUSTER_NM"\" - |
yq e '.spec.selector.matchLabels.app.kubernetes.io/name = "'"$CLUSTER_NM"\" - |
yq e '.spec.template.metadata.labels.app.kubernetes.io/name = "'"$CLUSTER_NM"\" - |
yq e '.spec.replicas = '"$N_NODES" - |
yq e '.spec.template.spec.volumes[1].secret.secretName = "'"$SECRET_NM"\" - |
yq e '.spec.template.spec.containers[0].image = "'"$IMAGE"\" - |
yq e '.spec.template.spec.volumes[3].persistentVolumeClaim.claimName = "'"$MOUNT_NAME"\" - > $WD/k8s_spec_temp.yml

# Template k8 configuration - shared mount
cat $WD/k8_spec_ssd-cluster.yml |
yq e '.metadata.name = "'"$MOUNT_NAME"\" - > $WD/k8_spec_ssd-cluster_temp.yml

# Delete previous and setup deployment
kubectl delete statefulsets/$CLUSTER_NM || { echo 'No previous cluster'; }
kubectl delete persistentvolumeclaims/$MOUNT_NAME || { echo 'No previous mount'; }

kubectl apply -f $WD/k8_spec_ssd-cluster_temp.yml
kubectl apply -f $WD/k8s_spec_temp.yml

echo Waiting for cluster deployment to complete...
kubectl rollout status --watch --timeout=600s statefulsets/$CLUSTER_NM  || { echo 'Cluster deployment failed' ; exit 1; }

echo Generate hosts file
kubectl get pods -o wide | grep $CLUSTER_NM | awk '{print $6 " slots=8"}' > $WD/hostfile
cat $WD/hostfile | cut -f1 -d' ' > $WD/hosts
export MAIN_ID=$(kubectl get pods | grep $CLUSTER_NM | awk '{print $1}' | head -n 1)

echo Copying ssh key and host file to main node:
echo $MAIN_ID
kubectl cp $WD/hostfile $MAIN_ID:/job
kubectl cp $WD/hosts $MAIN_ID:/job
kubectl cp $WD/id_rsa $MAIN_ID:/home/mchorse/.ssh

rm $WD/id_rsa* $WD/hostfile $WD/hosts $WD/k8s_spec_temp.yml $WD/k8_spec_ssd-cluster_temp.yml $WD/post_start_script.sh

echo Remote shell into main $MAIN_ID
kubectl exec --stdin --tty $MAIN_ID -- /bin/bash -c "cd /home/mchorse/gpt-neox; bash"
