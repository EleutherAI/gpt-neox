import random
import deepspeed
import torch
from torch.utils.data import DataLoader
from tqdm.auto import trange
import torch.distributed as distributed

from gpt_neox import (GPTNeoX_Pipe, AutoregressiveWrapper, GPT2Dataset, extract_tarfile,
                      prepare_optimizer_parameters, get_tokenizer, is_main, prepare_data)

from gpt_neox.utils import get_args, get_params

import GPUtil

# arguments
train_args = get_args()
params = get_params(train_args.model)

# tokenizer
tokenizer = get_tokenizer(tokenizer_type=params["tokenizer"].get("type", None),
                          from_pretrained=params["tokenizer"].get("from_pretrained", True),
                          add_padding_token=params["tokenizer"].get("add_padding_token", False))
vocab_size = len(tokenizer) if params["vocab_size"] is None else params["vocab_size"]

# model
deepspeed.init_distributed(dist_backend='nccl')
torch.distributed.barrier()  # barrier will force processes to stop until *all* processes have reached the barrier

def loss_function(x, y):
    losses = torch.nn.functional.cross_entropy(x, y, reduction='none')
    loss = losses.mean()
    return loss
        
model = gpt_neox.GPTNeoX_Pipe(
    num_tokens=params["vocab_size"],
    dim=params["hidden_dim"],
    seq_len=params["seq_len"],
    depth=params["n_layers"],
    heads=params["n_heads"],
    dim_head=params["dim_head"],
    loss_fn = loss_function,#torch.nn.CrossEntropyLoss(),
    num_stages = params.get("pipeline_num_stages", 2)
)
model = AutoregressiveWrapper(model)

# optimizer
ds_model_params = prepare_optimizer_parameters(model)
optim = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

# deepspeed loader
model_engine, optim, train_loader, _ = deepspeed.initialize(args=train_args,
                                                            model=model,
                                                            optimizer=optim,
                                                            model_parameters=ds_model_params,
                                                            training_data=train_dataset)

# prepare data
dset_params = params["dataset"]
assert dset_params is not None

if is_main(train_args):
    prepare_data(dset_params["name"])
    torch.distributed.barrier()  # barrier will force processes to stop until *all* processes have reached the barrier
else:
    torch.distributed.barrier()
    
# data loading
train_dataset = GPT2Dataset(glob_pattern=dset_params["train_path"],
                            seq_len=params["seq_len"],
                            train=True,
                            **dset_params)
train_loader = model_engine.deepspeed_io(train_dataset, pin_memory=params.get("pin_memory", False))

eval_dataset = GPT2Dataset(glob_pattern=dset_params["eval_path"],
                           seq_len=params["seq_len"],
                           train=False,
                           **dset_params)

val_loader = DataLoader(eval_dataset, batch_size=params["eval_batch_size"])
val_loader = iter(val_loader)


for step, batch in enumerate(data_loader):
    #forward() method
    loss = model_engine(batch)

    #runs backpropagation
    model_engine.backward(loss)

    #weight update
    model_engine.step()
