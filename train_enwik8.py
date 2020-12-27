from gpt_neox import (GPTNeoX, AutoregressiveWrapper, TextSamplerDataset,
                      cycle, prepare_optimizer_parameters, decode_tokens, prepare_enwik8_data)
import random
import torch
from torch.utils.data import DataLoader
import deepspeed
from tqdm.auto import trange
import argparse
import json
from collections import defaultdict


def get_args():
    parser = argparse.ArgumentParser(description='GPTNeox Deepspeed Training Script')
    # Include DeepSpeed configuration arguments
    parser.add_argument('--model', type=str, default="base_model")
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

# instantiate GPT-like decoder model
model = GPTNeoX(
    num_tokens=params["VOCAB_SIZE"],
    dim=params["HIDDEN_DIM"],
    seq_len=params["SEQ_LEN"],
    depth=params["N_LAYERS"],
    heads=params["N_HEADS"],
    dim_head=params["DIM_HEAD"]
)

model = AutoregressiveWrapper(model)
model.cuda()

# prepare enwik8 data
data_train, data_val = prepare_enwik8_data()
train_dataset = TextSamplerDataset(data_train, params["SEQ_LEN"])
val_dataset = TextSamplerDataset(data_val, params["SEQ_LEN"])
val_loader = cycle(DataLoader(val_dataset, batch_size=params["BATCH_SIZE"]))

# optimizer
optim = torch.optim.Adam(model.parameters(), lr=params["LEARNING_RATE"])

# training
ds_model_params = prepare_optimizer_parameters(model)

# deepspeed loader
model_engine, optim, train_loader, _ = deepspeed.initialize(args=train_args,
                                                            model=model,
                                                            optimizer=optim,
                                                            model_parameters=ds_model_params,
                                                     training_data=train_dataset)

pbar = trange(params["NUM_EPOCHS"], mininterval=10., desc='Training Model', dynamic_ncols=True)
for _ in pbar:
    for i, data in enumerate(train_loader):
        model_engine.train()
        is_main = model_engine.local_rank == 0
        data = data.to(model_engine.local_rank)

        loss = model_engine(data)
        model_engine.backward(loss)
        model_engine.step()

        pbar.set_description(f'Training Loss: {loss.item():.4f}')
        pbar.update()

        if is_main and i % params["VALIDATE_EVERY"] == 0:
            model.eval()
            with torch.no_grad():
                val_data = next(val_loader).cuda()
                loss = model(val_data)
                pbar.write(f'Validation Loss: {loss.item()}')

        if is_main and i % params["GENERATE_EVERY"] == 0:
            model.eval()
            inp = random.choice(val_dataset)[:-1]
            prime = decode_tokens(inp)
            pbar.write(f"{prime} \n\n {'*' * 100}")
            sample = model.generate(inp.cuda(), params["GENERATE_LENGTH"])
            output_str = decode_tokens(sample)
            pbar.write(output_str)
