import os
import sys
sys.path.append('.')
from megatron.model.image_prefix import ImagePrefix
from megatron.neox_arguments import NeoXArgs


neox_args = NeoXArgs.from_ymls(['./configs/magma_pythia_410M.yml', './configs/magma_setup.yml'])
image_prefix = ImagePrefix(
    config = neox_args,
    out_dim=neox_args.hidden_size,
)
print(f"Downloaded pretrain weight of {neox_args.encoder_name} to {neox_args.load} !")
