import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_script", type=str, default="small.yml",
                    help="the script to run, defaults to small.yml")
args = parser.parse_args()

def run(run_script):
    command = f"NCCL_P2P_LEVEL=2 ./deepy.py pretrain_gpt2.py -d configs eleutherai_cluster.yml {args.train_script}"
    os.system(command)

if __name__ == "__main__":
    run()
