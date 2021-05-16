from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import os
import itertools

inp = 'configs/medium.yml'
out = 'configs/tmp.yml'

gridsearch_params = {'rotary_pct': [0.01, 0.25, 0.5, 0.75, 1.0],
                     'rotary_emb_base': [1000, 10000, 100000]
                    }

keys, values = zip(*gridsearch_params.items())
for v in itertools.product(*values):
        experiment = dict(zip(keys, v))

        # make tmp yml with selected params changed
        with open(inp, 'r') as f:
                data = load(f, Loader=Loader)
        data.update(experiment)
        with open(out, 'w') as f:
                f.write(dump(data))

        # sync configs and start run
        os.system('bash tools/syncdir.sh configs')
        os.system('./deepy.py pretrain_gpt2.py -d configs tmp.yml eleutherai_cluster.yml')

        # kill any hanging processes and delete saved checkpoints
        os.system("pdsh -f 1024 -R ssh -w ^/job/hosts 'pkill -f deepy.py'")
        os.system('rm -rf /mnt/ssd-cluster/checkpoints')
