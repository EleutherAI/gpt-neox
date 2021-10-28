pdsh -f 1024 -R ssh -w ^/job/hosts 'pkill -f train.py'
