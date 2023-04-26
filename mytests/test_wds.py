import sys
sys.path.append('/home/lfsm/code/gpt-neox/')
from megatron.neox_arguments import NeoXArgs

ymls = ['/home/lfsm/code/gpt-neox/configs/20B.yml']

neox_args = NeoXArgs.from_ymls(ymls)
neox_args.build_tokenizer()

from megatron.data.webdataset import get_wds_data
data = get_wds_data(neox_args)

