for id in $(kubectl get pods | grep eleuther-neox | awk '{print $1}')
do
    echo running command "$@" on container $id
    echo "$@" | kubectl exec --stdin $id -- /bin/bash
done
