import torch
import mpu
import time
import os
from gpt_neox.mpu_loading import get_batch

def generate_samples(model, tokenizer, device,generate_length=2048,seq_length=2048):
    
    context_count=0
    model.eval()
    with torch.no_grad():
        while True:
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            terminate_runs=0

            if mpu.get_model_parallel_rank() == 0:
                raw_text = input("\nContext prompt (stop to exit) >>> ")
                while not raw_text:
                    print('Prompt should not be empty!')
                    raw_text = input("\nContext prompt (stop to exit) >>> ")
           
                if "stop" in raw_text:
                    terminate_runs = 1
                else:
                    context_tokens = tokenizer(raw_text, max_length=seq_length, return_tensors='pt',\
                padding='max_length', truncation=True)['input_ids']
                    context_length = len(context_tokens)

            else:
                context_tokens = tokenizer(raw_text, max_length=seq_length, return_tensors='pt',\
                padding='max_length', truncation=True)['input_ids']
                context_length = len(context_tokens)
            
            terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
            torch.distributed.broadcast(terminate_runs_tensor, mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())
            terminate_runs = terminate_runs_tensor[0].item()

            if terminate_runs == 1:
                return

            
            context_tokens_tensor = torch.cuda.LongTensor(context_tokens.to(device))
            context_tokens_tensor = context_tokens_tensor.unsqueeze(0)
            print(context_tokens_tensor.shape)
            context_length_tensor = torch.cuda.LongTensor([context_length])

            torch.distributed.broadcast(context_length_tensor, mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())
            torch.distributed.broadcast(context_tokens_tensor, mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())

            context_length = context_length_tensor[0].item()
    
            tokens,attention_mask,position_ids = get_batch(context_tokens=context_tokens_tensor,eod_token=tokenizer.eos_token_id)

            start_time = time.time()

            sample = model.generate(tokens, generate_length,attention_mask=attention_mask,position_ids=position_ids)
            output_str = tokenizer.decode(sample)
                
            if mpu.get_model_parallel_rank() == 0:
                os.system('clear')
                print("\nTaken time {:.2f}\n".format(time.time() - start_time), flush=True)
                print("\nContext:", raw_text, flush=True)
                print("\nGPT:", output_str, flush=True)
            raw_text = None

            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            context_count += 1
