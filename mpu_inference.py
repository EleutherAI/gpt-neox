import random
import deepspeed
import torch
from torch.utils.data import DataLoader
from tqdm.auto import trange
import torch.distributed as distributed

from gpt_neox import (GPTNeoX, AutoregressiveWrapper, TFRecordDataset, extract_tarfile,
                      prepare_optimizer_parameters, get_tokenizer, is_main, prepare_data)

from gpt_neox.utils import get_args, get_params, load_checkpoint

from gpt_neox.mpu_loading import get_model, set_random_seed
from gpt_neox.mpu_generation import generate_samples


import mpu
import os

seed = 7
train_args = get_args()
params = get_params(train_args.model)

# tokenizer
tokenizer = get_tokenizer(tokenizer_type=None,
                          from_pretrained=True,
                          add_padding_token=True)
vocab_size = len(tokenizer) if params["vocab_size"] is None else params["vocab_size"]

# prepare data
dset_params = params["dataset"]
assert dset_params is not None

#initialize distributed for mpu
deepspeed.init_distributed(dist_backend='nccl')

if train_args.local_rank is not None:
    device = train_args.local_rank
    torch.cuda.set_device(device)

# Set the model-parallel / data-parallel communicators.
model_parallel_size = 1
world_size = int(os.getenv("WORLD_SIZE", '1'))
model_parallel_size = min(model_parallel_size, world_size)
mpu.initialize_model_parallel(model_parallel_size)
set_random_seed(seed)


torch.distributed.barrier()  # barrier will force processes to stop until *all* processes have reached the barrier
if is_main(train_args):
    prepare_data(dset_params["name"])
    torch.distributed.barrier()  # barrier will force processes to stop until *all* processes have reached the barrier
else:
    torch.distributed.barrier()

# instantiate GPT-like decoder model
model = get_model(vocab_size,params)
if 'load' in vars(train_args):
    _ = load_checkpoint(model, None, None, train_args)

model = AutoregressiveWrapper(model,seq_len=params['seq_len'])

generate_samples(model,tokenizer,device)



