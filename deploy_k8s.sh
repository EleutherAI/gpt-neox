kubectl delete deploy/eleuther-neox
kubectl apply -f kubernetes/deploy_k8s.yml
echo Waiting for deploy to complete...
kubectl wait --for=condition=available --timeout=600s deployment/eleuther-neox || exit

kubectl get pods -o wide | grep eleuther-neox | awk '{print $6 " slots=8"}' > hosts
export MASTER_ID=$(kubectl get pods | grep eleuther-neox | awk '{print $1}' | head -n 1)
echo $MASTER_ID
kubectl cp $PWD/hosts $MASTER_ID:/app
#echo 'git remote set-url origin https://github.com/EleutherAI/gpt-neox/ && git pull' | kubectl exec --stdin --tty $MASTER_ID -- /bin/bash
echo "$@" | kubectl exec --stdin --tty $MASTER_ID -- /bin/bash
