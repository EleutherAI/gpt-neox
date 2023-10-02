# Bash Scripts
Useful for running distributed per-node scripts on e.g. Kubernetes 

* `kill.sh` kills all python processes
* `killall.sh` uses pdsh to kill all `train.py` processes on the nodes listed in `/job/hosts/`
* `sync_cmd.sh` uses pdsh to run a command on all the nodes listed in `/job/hosts/`
* `sync.sh` uses pdcp to copy every file in a provided path to all of the nodes listed in `/job/hosts/`
* `syncdir.sh` uses pdcp to copy every file in a provided path to all of the nodes listed in `/job/hosts/`
