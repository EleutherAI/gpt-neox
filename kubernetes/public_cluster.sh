#!/usr/bin/env bash

#  --- USAGE ---
# $ public_cluster.sh [name_suffix=$USER]
# You need to install yq

# Check yq
yq &> /dev/null || { echo 'You need to install `yq >= v4`. `brew install yq` or `pip install yq`' ; exit 1; }

if [ -z "$1" ]
  then
    echo "No argument supplied: you must provide a cluster name suffix"
    exit 1
fi

SUFFIX=${1}

CLUSTER_NM='neox-'"$SUFFIX"
WD=`dirname "$BASH_SOURCE"`

echo Getting main node
export MAIN_ID=$(kubectl get pods | grep $CLUSTER_NM | awk '{print $1}' | head -n 1)
if [ -z "$MAIN_ID" ]; then
    echo "Bad cluster name. Couldn't obtain main node"
    exit 1
fi
echo Main node: $MAIN_ID

echo MAKING CLUSTER $CLUSTER_NM PUBLIC
echo Generating random password:
echo "==================="
PASSWORD=$(openssl rand -base64 32)
echo $PASSWORD
echo "==================="
echo Save ^^ PW as otherwise will be lost

echo Chaning password on main node
pw_cmd="sudo usermod --password \$(echo $PASSWORD | openssl passwd -1 -stdin) mchorse"
kubectl exec --stdin --tty $MAIN_ID -- /bin/bash -c "$pw_cmd"

SERVICE_NM="$CLUSTER_NM-service"
cat $WD/pubic-IP-service.yaml |
yq e '.metadata.name = "'"$SERVICE_NM"\" - |
yq e '.spec.selector["statefulset.kubernetes.io/pod-name"] = "'"$MAIN_ID"\" - > $WD/pubic-IP-service_temp.yml

kubectl delete service/$SERVICE_NM

echo Setting up Public IP. Service $SERVICE_NM
kubectl apply -f $WD/pubic-IP-service_temp.yml

PUBLIC_ID=$(kubectl get service/$SERVICE_NM -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo Public IP $PUBLIC_ID
echo ssh command:
echo ssh mchorse@$PUBLIC_ID