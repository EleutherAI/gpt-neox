import sys
sys.path.append('/css/home/lfsm/code/gpt-neox')
from megatron.neox_arguments import NeoXArgs

ymls = ['/home/lfsm/code/gpt-neox/configs/800M.yml', '/home/lfsm/code/gpt-neox/configs/local_setup.yml']

args_loaded = NeoXArgs.from_ymls(ymls)
deepspeed_main_args = args_loaded.get_deepspeed_main_args()
print(deepspeed_main_args)
