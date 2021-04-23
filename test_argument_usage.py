import os
import re
import sys

sys.path.append('./megatron/')
from neox_arguments import NeoXArgs

def check_file(file):
    with open(file, 'r') as f:
        text = f.read()
    matches = re.findall("(?<=args\.).{2,}?(?=[\s\n(){}+-/*;:,=])", text)
    return list(dict.fromkeys(matches))

def run_test(file):
    neox_args = list(NeoXArgs.__dataclass_fields__)
    missing = []
    matches = check_file(file)
    for match in matches:
        if match not in neox_args:
            missing.append(match)
    return missing

if __name__ == "__main__":

    files = [] 
    foldersToCheck = ['./megatron/'] 
    while (len(foldersToCheck) > 0): 
        for (dirpath, dirnames, filenames) in os.walk(foldersToCheck[0]): 
            while(len(dirnames) > 0): 
                foldersToCheck.append(foldersToCheck[0] + dirnames[0] + "/") 
                del dirnames[0] 
            while(len(filenames) > 0): 
                if filenames[0].endswith('py'):
                    files.append(foldersToCheck[0] + filenames[0]) 
                
                del filenames[0] 
            del foldersToCheck[0] 

    files.remove('./megatron/config_monster.py')
    files.remove('./megatron/text_generation_utils.py')
    files.remove('./megatron/arguments.py')

    for file in files:
        out = run_test(file)
        if out != []:
            print(f"{file}: {out}")
    
    #print(f"training: {run_test('./megatron/training.py')}")
    #print(f"global_vars: {run_test('./megatron/global_vars.py')}")
    #print(f"initialize: {run_test('./megatron/initialize.py')}")
    #print(f"learning_rates: {run_test('./megatron/learning_rates.py')}")
    #print(f"logging: {run_test('./megatron/logging.py')}")
    #print(f"memory: {run_test('./megatron/memory.py')}")
    #print(f"module: {run_test('./megatron/module.py')}")
    #print(f"optimizers: {run_test('./megatron/optimizers.py')}")
    #print(f"generation: {run_test('./megatron/text_generation_utils.py')}")
    #print(f"utils: {run_test('./megatron/utils.py')}")

    ['--num_gpus', '2', 
    'pretrain_gpt2.py', 
    '--num-layers', '12', 
    '--hidden-size', '768', 
    '--num-attention-heads', '12', 
    '--max-position-embeddings', '2048', 
    '--attention-dropout', '0', 
    '--hidden-dropout', '0', 
    '--weight-decay', '0', 
    '--batch-size', '4', 
    '--checkpoint-activations', 
    '--checkpoint-num-layers', '1', 
    '--train-iters', '320000', 
    '--log-interval', '100', 
    '--tensorboard-dir', 'tensorboard', 
    '--no-weight-tying', 
    '--pos-emb', 'none', 
    '--norm', 'rmsnorm', 
    '--lr-decay-style', 'cosine', 
    '--lr-decay-iters', '320000', 
    '--warmup', '0.01', 
    '--save', 'checkpoints', 
    '--save-interval', '10000', 
    '--keep-last-n-checkpoints', '4', 
    '--load', 'checkpoints', 
    '--model-parallel-size', '1', 
    '--pipe-parallel-size', '1', 
    '--distributed-backend', 'nccl', 
    '--eval-iters', '10', 
    '--eval-interval', '1000', 
    '--data-path', 'data/enron/enron_text_document', 
    '--split', '949,50,1', 
    '--vocab-file', 
    'data/gpt2-vocab.json', 
    '--merge-file', 
    'data/gpt2-merges.txt', 
    '--seq-length', '2048', 
    '--data-impl', 'mmap', 
    '--log-dir', 'logs', 
    '--partition-activations', 
    '--synchronize-each-layer', 
    '--wandb_group', '6Vr9eNVMQrADkqMUvMeQnf', 
    '--git_hash', 'efd39e5', 
    '--deepspeed', 
    '--fp16', 
    '--gas', '1', 
    '--zero-stage', '0', 
    '--zero-reduce-scatter', 
    '--zero-contiguous-gradients', 
    '--zero-reduce-bucket-size', '500000000', 
    '--zero-allgather-bucket-size', '500000000', 
    '--clip-grad', '1.0', 
    '--lr', '0.0006', 
    '--adam-beta1', '0.9',
    '--adam-beta2', '0.999', 
    '--adam-eps', '1e-08', 
    '--momentum', '0.0', 
    '--deepspeed_config', '{"train_batch_size":8.0,"train_micro_batch_size_per_gpu":4,"gradient_accumulation_steps":1,"optimizer":{"type":"Adam","params":{"lr":0.0006,"max_grad_norm":1.0,"betas":[0.9,0.999]}},"fp16":{"enabled":true,"loss_scale":0,"loss_scale_window":1000,"hysteresis":2,"min_loss_scale":1},"gradient_clipping":1.0,"zero_optimization":{"stage":0,"allgather_partitions":true,"allgather_bucket_size":500000000,"overlap_comm":true,"reduce_scatter":true,"reduce_bucket_size":500000000,"contiguous_gradients":true,"cpu_offload":false},"steps_per_print":10,"wall_clock_breakdown":true,"deepspeed":true}']