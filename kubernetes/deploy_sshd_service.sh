# replace 'sid' in .yaml files with some unique name

kubectl apply -f sshd-root-pvc.yaml
kubectl apply -f sshd-data-pvc.yaml
kubectl apply -f sshd-service.yaml
kubectl apply -f sshd-deployment.yaml

echo "getting pod id"

kubectl get pods

echo "run 'kubectl logs -f <pod ID> init' to get root pw"
echo "then 'kubectl get service' to get the SSH service external IP"