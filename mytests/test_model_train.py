import os
import sys
import torch
sys.path.append('/ccs/home/lfsm/code/gpt-neox/')
import torch
import torch.nn as nn
from megatron.neox_arguments import NeoXArgs
from megatron.initialize import initialize_megatron
from megatron import mpu
import deepspeed
from megatron.model import GPT2ModelPipe
from megatron.training import setup_model_and_optimizer
from megatron.data.data_utils import build_web_train_valid_test_data_iterators

torch.manual_seed(7)

neox_args = NeoXArgs.from_ymls(['/ccs/home/lfsm/code/gpt-neox/configs/19M.yml', '/ccs/home/lfsm/code/gpt-neox/configs/local_setup.yml'])
#neox_args = NeoXArgs.from_ymls(['/home/lfsm/code/gpt-neox/configs/20B.yml'])
neox_args.configure_distributed_args()
neox_args.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab

neox_args.train_data_paths = r'/gpfs/alpine/csc499/proj-shared/LAION-400m-webdataset/data/{41300..41401}.tar'
neox_args.valid_data_paths = r'/gpfs/alpine/csc499/proj-shared/LAION-400m-webdataset/data/{41000..41101}.tar'

initialize_megatron(neox_args=neox_args)
model, optimizer, lr_scheduler = setup_model_and_optimizer(neox_args)
train_data_iterator, valid_data_iterator, test_data_iterator = build_web_train_valid_test_data_iterators(neox_args)

# forward test
for i in range(10):
    loss = model.train_batch(data_iter=train_data_iterator)
    print(loss)
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
