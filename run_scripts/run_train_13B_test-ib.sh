module load cuda/11.8
python ./deepy.py train.py -d jarvis_configs/13B/test-ib params.yml setup.yml -H jarvis_configs/13B/test-ib/13B_test-ib_hostfile

