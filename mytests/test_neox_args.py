import sys
sys.path.append('/home/lfsm/code/gpt-neox/')
from megatron.neox_arguments import NeoXArgs

ymls = ['/home/lfsm/code/gpt-neox/configs/800M.yml', '/home/lfsm/code/gpt-neox/configs/local_setup.yml']

args_loaded = NeoXArgs.from_ymls(ymls)

print(args_loaded)