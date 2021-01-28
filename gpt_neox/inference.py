import random
import deepspeed
import torch
from torch.utils.data import DataLoader
from tqdm.auto import trange
import torch.distributed as distributed

from gpt_neox import (GPTNeoX, AutoregressiveWrapper, GPT2Dataset, extract_tarfile,
                      prepare_optimizer_parameters, get_tokenizer, is_main, prepare_data)

from gpt_neox.utils import get_args, get_params


train_args = get_args()
params = get_params(train_args.model)

# tokenizer
tokenizer = get_tokenizer(tokenizer_type=params["tokenizer"].get("type", None),
                          from_pretrained=params["tokenizer"].get("from_pretrained", True),
                          add_padding_token=params["tokenizer"].get("add_padding_token", False))
vocab_size = len(tokenizer) if params["vocab_size"] is None else params["vocab_size"]

# instantiate GPT-like decoder model
model = GPTNeoX(
    num_tokens=vocab_size,
    dim=params["hidden_dim"],
    seq_len=params["seq_len"],
    depth=params["n_layers"],
    heads=params["n_heads"],
    dim_head=params["dim_head"],
    gradient_checkpointing=params.get("gradient_checkpointing", True)
)

model = AutoregressiveWrapper(model)
# prepare data
dset_params = params["dataset"]
assert dset_params is not None

deepspeed.init_distributed(dist_backend='nccl')
torch.distributed.barrier()  # barrier will force processes to stop until *all* processes have reached the barrier
if is_main(train_args):
    prepare_data(dset_params["name"])
    torch.distributed.barrier()  # barrier will force processes to stop until *all* processes have reached the barrier
else:
    torch.distributed.barrier()

eval_dataset = GPT2Dataset(glob_pattern=dset_params["eval_path"],
                           seq_len=params["seq_len"],
                           train=False,
                           **dset_params)