import torch
import mii
from transformers import pipeline
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, default='EleutherAI/pythia-160m', help='hf model name')
parser.add_argument('--trials', type=int, default=50, help='number of trials')
parser.add_argument('--dtype', type=str, default='fp16', help='Data type for model')
parser.add_argument('--tensor_parallel', type=int, default=1, help='Tensor parallelism degree')
parser.add_argument('--load_with_sys_mem', action='store_true', help='Load model with system memory')
args = parser.parse_args()

def hf_infer(model, torch_dtype, query=['Deepspeed is', 'Seattle is'], trials=1):

    generator = pipeline('text-generation', model=model, device=0, torch_dtype=torch_dtype)
    eos_token = generator.tokenizer.eos_token_id

    start_time = time.time()
    for i in range(trials):
        hf_result = generator(query, max_new_tokens=100, pad_token_id=eos_token)
    end_time = time.time()

    hf_time = (end_time - start_time) / trials

    generator = None
    torch.cuda.empty_cache()

    return eos_token, hf_result, hf_time
    
def mii_infer(model, eos_token, query=['Deepspeed is', 'Seattle is'], trials=1):
    generator = mii.mii_query_handle(model + '_deploy')
    start_time = time.time()
    for i in range(trials):
        mii_result = generator.query({'query': query}, pad_token_id=eos_token, max_new_tokens=100)
    end_time = time.time()
    mii_time = (end_time - start_time) / trials

    return mii_result, mii_time

def main():

    dtype_mapping = {
    'fp16': torch.float16,
    'fp32': torch.float32,
    'fp64': torch.float64,
    'int8': torch.int8,
    'int16': torch.int16,
    'int32': torch.int32,
    'int64': torch.int64
    }

    torch_dtype = dtype_mapping[args.dtype]
    load_with_sys_mem = args.load_with_sys_mem
    tensor_parallel = args.tensor_parallel
    trials = args.trials
    model = args.model

    eos_token, hf_result, hf_time = hf_infer(model, torch_dtype, trials=trials)

    mii_configs = {'tensor_parallel': tensor_parallel, 'dtype': torch_dtype, 'load_with_sys_mem': load_with_sys_mem}
    mii.deploy(task='text-generation',
            model=model,
            deployment_name=model + '_deploy',
            mii_config=mii_configs)
    mii_result, mii_time = mii_infer(model, eos_token, trials=trials)

    print('HF sample output', hf_result)
    print('HF Average Inference time: ', hf_time)
    
    print('MII sample output', mii_result)
    print('MII Average Inference time: ', mii_time)

    mii.terminate(model + '_deploy')

if __name__ == '__main__':
    main()
