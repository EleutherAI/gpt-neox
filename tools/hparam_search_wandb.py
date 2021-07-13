import os
os.system("NCCL_P2P_LEVEL=2 ./deepy.py pretrain_gpt2.py -d configs eleutherai_cluster.yml small.yml")
