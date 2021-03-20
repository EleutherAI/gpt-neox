#!/usr/bin/env bash

#  --- USAGE ---
# $ deploy_k8.sh [branch=main] [n_nodes=4] [name_suffix=$USER] [A100s='no'. Option:no/yes] [image]
# You need to install yq

# Check yq
yq &> /dev/null || { echo 'You need to install `yq >= v4`. `brew install yq` or `pip install yq`' ; exit 1; }

WD_BRANCH=$(git branch  --no-color --show-current)
WD_BRANCH="${WD_BRANCH/\//-}"  # remove forward slashes and replace with underscore
DEFAULT_IMAGE="leogao2/gpt-neox:sha-86664f3"

BRANCH=${1:-main}
N_NODES=${2:-4}
SUFFIX=${3:-$(whoami)}
USE_A100s=${4:-"no"}
IMAGE=${5:-$DEFAULT_IMAGE}

CLUSTER_NM='neox-'"$SUFFIX"
WD=`dirname "$BASH_SOURCE"`

# Use A100s? Default to no
if [ "$USE_A100s" = "yes" ]; then
    CLUSTER_SPEC=$WD/k8s_a100_cluster_spec.yml
    GPUS_PER_NODE=6
    awk_print='{print $6 " slots=6"}'
else
    USE_A100s="no"
    CLUSTER_SPEC=$WD/k8s_cluster_spec.yml
    GPUS_PER_NODE=8
    awk_print='{print $6 " slots=8"}'
fi

echo BRANCH $BRANCH. N-NODES $N_NODES. CLUSTER NAME $CLUSTER_NM. Use A100s: $USE_A100s. DOCKER IMAGE $IMAGE.

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

# This script is designed to work with any image that is debian based to allow for quick prototyping.
# The only requirement is `/bin/bash`. It works if the user is root or another such as `mchorse`.
# HINT: the `post_start_script` cannot have blank lines and NO comments.
# If this script hangs the pod will never enter a success state even though its running.
post_start_script="
export DEBIAN_FRONTEND=noninteractive;
apt-get update -y || { sleep 3; sudo apt-get update -y; };
apt-get install -y sudo pdsh git || { sleep 3; apt-get install -y pdsh git; };
if [ \$(dpkg-query -W -f='\${Status}' ssh 2>/dev/null | grep -c 'ok installed') -eq 0 ];
then
  sudo apt-get install ssh;
fi;
echo '    StrictHostKeyChecking no' >> ~/.ssh/config;
sudo mkdir -p /run/sshd;
sudo /usr/sbin/sshd;
sudo rm -rf /job;
sudo mkdir -p /job ~/.ssh;
USER=\$(whoami);
sudo chown \$USER:\$USER /job;
sudo cp /secrets/id_rsa.pub ~/.ssh/authorized_keys;
sudo chown -R \$USER:\$USER ~/.ssh;
sudo chown \$USER:\$USER ~/.ssh/authorized_keys;
chmod 600 ~/.ssh/authorized_keys;
chmod 700 ~/.ssh;
echo 'export LC_ALL=C.UTF-8' >> ~/.bashrc;
echo 'export LANG=C.UTF-8' >> ~/.bashrc;
cd ~;
git clone --branch $BRANCH https://github.com/EleutherAI/gpt-neox.git;
"
if [ -n "$WANDB_APIKEY" ]
then
      post_start_script+=" wandb login $WANDB_APIKEY || true; "
fi
post_start_script+="exit 0;" # Always exit with ok status

echo $post_start_script > $WD/post_start_script.sh

# Add ssh key to k8 secrets and post start script
DATE=$(date +%s)
SECRET_NM="$CLUSTER_NM-$DATE"
kubectl create secret generic $SECRET_NM \
  --from-file=id_rsa.pub=$WD/id_rsa.pub \
  --from-file=post_start_script.sh=$WD/post_start_script.sh

# Template k8 configuration - deployment
MOUNT_NAME="$CLUSTER_NM-ssd-cluster"
cat $CLUSTER_SPEC |
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

if [ "$USE_A100s" = "yes" ]; then
    kubectl get pods -o wide | grep $CLUSTER_NM | awk '{print $6 " slots=6"}' > $WD/hostfile
else
    kubectl get pods -o wide | grep $CLUSTER_NM | awk '{print $6 " slots=8"}' > $WD/hostfile
fi

cat $WD/hostfile | cut -f1 -d' ' > $WD/hosts
export MAIN_ID=$(kubectl get pods | grep $CLUSTER_NM | awk '{print $1}' | head -n 1)

echo Copying ssh key and host file to main node:
echo $MAIN_ID
HOME_DIR=$(kubectl exec $MAIN_ID -- /bin/bash -c 'cd ~; pwd')
kubectl cp $WD/hostfile $MAIN_ID:/job
kubectl cp $WD/hosts $MAIN_ID:/job
kubectl cp $WD/id_rsa $MAIN_ID:$HOME_DIR/.ssh

rm $WD/id_rsa* $WD/hostfile $WD/hosts $WD/k8s_spec_temp.yml $WD/k8_spec_ssd-cluster_temp.yml $WD/post_start_script.sh

echo Remote shell into main $MAIN_ID
kubectl exec --stdin --tty $MAIN_ID -- /bin/bash -c "cd $HOME_DIR/gpt-neox; bash"
