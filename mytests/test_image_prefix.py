import os
import sys
sys.path.append('/gpfs/alpine/csc499/scratch/kshitijkg/magma/gpt-neox')

from megatron.model.image_prefix import ImagePrefix
from megatron.neox_arguments import NeoXArgs

neox_args = NeoXArgs.from_ymls(['/gpfs/alpine/csc499/scratch/kshitijkg/magma/gpt-neox/configs/pythia_410M.yml', '/gpfs/alpine/csc499/scratch/kshitijkg/magma/gpt-neox/configs/summit_setup.yml'])

image_prefix = ImagePrefix(
    config = neox_args,
    out_dim=neox_args.hidden_size,
)

print("hello world")
