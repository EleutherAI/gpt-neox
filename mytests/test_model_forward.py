import os
import sys
import torch
sys.path.append('/home/lfsm/code/gpt-neox/')
import torch
import torch.nn as nn
from megatron.neox_arguments import NeoXArgs
from megatron.initialize import initialize_megatron
from megatron import mpu
import deepspeed
from megatron.model import GPT2ModelPipe
from megatron.training import setup_model_and_optimizer

torch.manual_seed(7)

neox_args = NeoXArgs.from_ymls(['/home/lfsm/code/gpt-neox/configs/800M.yml', '/home/lfsm/code/gpt-neox/configs/local_setup.yml'])
neox_args.configure_distributed_args()
neox_args.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab
initialize_megatron(neox_args=neox_args)

model, optimizer, lr_scheduler = setup_model_and_optimizer(neox_args)

# pesudo dataset

# data_list = list()
# context_tokens_tensor = torch.randint(
#     0, 10000, (4, 2048 + 1)
# ).to(torch.int64)
# for i in range(10):
#     data_list.append({"text": context_tokens_tensor.clone()})
# data_iterator = iter(data_list)

mbs = 4
data_list = list()
images = torch.ones(mbs,3,224,224).half()
captions = torch.ones(mbs,2048).to(torch.int64)
for i in range(10):
    # data_list.append({"text":captions.clone()})
    data_list.append({'img':images.clone(),'text':captions.clone()})
    # data_list.append([images.clone(),captions.clone()])
data_iterator = iter(data_list)

# forward test
for i in range(10):
    loss = model.train_batch(data_iter=data_iterator)
    print(loss)
    # loss = train_step(neox_args,timers,data_iterator,model,optimizer,lr_scheduler)
    print(f'finish {i} step')

"""

Pipeline:
    Train step:
        ({'lm_loss': tensor(11.0298, device='cuda:0')}, 0)
        finish 0 step

    train_batch:
        train_batch loss is 11.029833793640137
        finish 0 step

"""