import os
import sys
sys.path.append('/home/lfsm/code/gpt-neox/')

import torch
from megatron.neox_arguments import NeoXArgs
from megatron.initialize import initialize_megatron
from megatron.training import setup_model_and_optimizer

neox_args = NeoXArgs.from_ymls(['/home/lfsm/code/gpt-neox/configs/800M.yml', '/home/lfsm/code/gpt-neox/configs/summit_setup.yml'])
neox_args.configure_distributed_args()
neox_args.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab
initialize_megatron(neox_args=neox_args)
model, optimizer, lr_scheduler = setup_model_and_optimizer(neox_args)
memory_usage = torch.cuda.memory_allocated() / 1e9
print("Current GPU memory usage: {:.2f} GB".format(memory_usage))