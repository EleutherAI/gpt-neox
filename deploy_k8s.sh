kubectl delete deploy/eleuther-neox
kubectl apply -f kubernetes/deploy_k8s.yml
ssh-keygen -t rsa -f id_rsa -N ""
echo Waiting for deploy to complete...
kubectl wait --for=condition=available --timeout=600s deployment/eleuther-neox || exit

kubectl get pods -o wide | grep eleuther-neox | awk '{print $6 " slots=8"}' > hosts
export MASTER_ID=$(kubectl get pods | grep eleuther-neox | awk '{print $1}' | head -n 1)
echo $MASTER_ID
kubectl cp $PWD/hosts $MASTER_ID:/app
kubectl cp $PWD/id_rsa $MASTER_ID:/root/.ssh

mv id_rsa.pub authorized_keys

for id in $(kubectl get pods | grep eleuther-neox | awk '{print $1}')
do
    echo copying keys to $id
    kubectl cp $PWD/authorized_keys $id:/root/.ssh/
    echo 'chmod 600 ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chown -R root /root/.ssh' | kubectl exec --stdin $id -- /bin/bash
done
rm authorized_keys hosts
rm id_rsa*

kubectl exec --stdin --tty $MASTER_ID -- /bin/bash
