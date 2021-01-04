import argparse
import json
import random
from collections import defaultdict

import deepspeed
import torch
from torch.utils.data import DataLoader
from tqdm.auto import trange

from gpt_neox import (GPTNeoX, AutoregressiveWrapper, GPT2Dataset,
                      prepare_optimizer_parameters, get_tokenizer, download_dataset, get_all_files)


def get_args():
    parser = argparse.ArgumentParser(description='GPTNeox Deepspeed Training Script')
    # Include DeepSpeed configuration arguments
    parser.add_argument('--model', type=str, default="gpt3_small")
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def get_params(model):
    model_path = model if model.endswith(".json") else f"./configs/{model}.json"
    with open(model_path) as f:
        params = json.load(f)
    return defaultdict(lambda: None, params)


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
    dim_head=params["dim_head"]
)

model = AutoregressiveWrapper(model)

# prepare data
dset_params = params["dataset"]
assert dset_params is not None

download_dataset(dataset=params["name"], dataset_dir=params["dir"])
files = get_all_files(filetype=params["filetype"], files_dir=params["dir"])
# TODO: SPLIT?

train_dataset = GPT2Dataset(files=files,
                            seq_len=params["seq_len"],
                            train=True,
                            **dset_params)
eval_dataset = GPT2Dataset(files=files,
                           seq_len=params["seq_len"],
                           train=False,
                           **dset_params)

val_loader = DataLoader(eval_dataset, batch_size=params["eval_batch_size"])
val_loader = iter(val_loader)

# optimizer
optim = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

# training
ds_model_params = prepare_optimizer_parameters(model)

# deepspeed loader
model_engine, optim, train_loader, _ = deepspeed.initialize(args=train_args,
                                                            model=model,
                                                            optimizer=optim,
                                                            model_parameters=ds_model_params,
                                                            training_data=train_dataset)

pbar = trange(params.get("train_steps", 1), mininterval=10., desc='Training Model', dynamic_ncols=True)
for _ in pbar:
    for i, data in enumerate(train_loader):
        if i > params["train_steps"]:
            break
        model_engine.train()
        is_main = model_engine.local_rank == 0
        data = data.to(model_engine.local_rank)

        loss = model_engine(data)
        model_engine.backward(loss)
        model_engine.step()

        pbar.set_description(f'Training Loss: {loss.item():.4f}')
        pbar.update()

        if params.get("validate_every") is not None:
            if is_main and i % params["validate_every"] == 0:
                model_engine.eval()
                with torch.no_grad():
                    val_data = next(val_loader).cuda()
                    loss = model_engine(val_data)
                    pbar.write(f'Validation Loss: {loss.item()}')

        if params.get("generate_every") is not None:
            if is_main and i % params["generate_every"] == 0:
                model.eval()
                val_data = next(val_loader).cuda()
                inp = random.choice(val_data)[:-1]
                prime = tokenizer.decode(inp)
                pbar.write(f"{prime} \n\n {'*' * 100}")
                sample = model.generate(inp.cuda(), params["generate_length"])
                output_str = tokenizer.decode(sample)
                pbar.write(output_str)
