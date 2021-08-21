pdsh -f 1024 -R ssh -w ^/job/hosts 'pkill -f pretrain_gpt2.py'
