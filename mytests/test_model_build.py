import os
import sys
sys.path.append('/home/lfsm/code/gpt-neox/')
from tests.common import distributed_test, model_setup, parametrize

ymls = ['/home/lfsm/code/gpt-neox/configs/800M.yml', '/home/lfsm/code/gpt-neox/configs/local_setup.yml']
model, optimizer, lr_scheduler, args_loaded = model_setup(ymls)

print(model)