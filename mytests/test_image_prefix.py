import os
import sys
sys.path.append('/ccs/home/lfsm/code/gpt-neox/')

from megatron.model.image_prefix import ImagePrefix
from megatron.neox_arguments import NeoXArgs

neox_args = NeoXArgs.from_ymls(['/ccs/home/lfsm/code/gpt-neox/configs/800M.yml', '/ccs/home/lfsm/code/gpt-neox/configs/local_setup.yml'])

image_prefix = ImagePrefix(
    config = neox_args,
    out_dim=neox_args.hidden_size,
)

print("hello world")
