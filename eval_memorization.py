#!/usr/bin/env python3
# coding=utf-8
import os
import logging
import time
import datetime
import torch
import copy
import boto3
import numpy as np
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from megatron.data.data_utils import build_the_dataset
from transformers import AutoModelForCausalLM
import transformers.utils as transformer_utils
import multiprocessing as mp
from tqdm import trange

def generate_dataset(batch_size, start_token_idx, end_token_idx, mp_queue, prefetch_min = 128):
    logging.info("Loading dataset")
    prefix = '/fsx/pile/pile_20B_tokenizer_text_document'

    if os.environ['MODEL'].endswith('deduped'):
        prefix = '/fsx/pile_deduped/pile_0.87_deduped_text_document'
    dataset = build_the_dataset(
        data_prefix = prefix,
        name = 'train_0',
        data_impl='mmap',
        num_samples=131727360,
        seq_length=2048,
        seed=1234,
        skip_warmup=True,
        build_index_mappings=False
    )

    idx_path = "/fsx/pile/pile_20B_tokenizer_text_document_test_0_indexmap_10292ns"
    if os.environ['MODEL'].endswith("deduped"):
        idx_path = "/fsx/pile_deduped/pile_0.87_deduped_text_document_train_0_indexmap_147164160ns"

    logging.info(f"using idx path: {idx_path}")
    dataset.doc_idx = np.load(f"{idx_path}_2048sl_1234s_doc_idx.npy")
    dataset.sample_idx = np.load(f"{idx_path}_2048sl_1234s_sample_idx.npy")
    dataset.shuffle_idx = np.load(f"{idx_path}_2048sl_1234s_shuffle_idx.npy")

    dataset.shuffle_idx_len = dataset.shuffle_idx.shape[0] - 1
    dataset.sample_idx_len = dataset.sample_idx.shape[0] - 1
    context_tokens = []
    true_continuation = []
    i = 0
    for i in range(start_token_idx, end_token_idx + 1):
        context_tokens.append(dataset[i]['text'][:32].tolist())
        true_continuation.append(dataset[i]['text'][32:64].tolist())

        if len(context_tokens) == batch_size:
            mp_queue.put((i - len(context_tokens) + 1, context_tokens, true_continuation))
            context_tokens = []
            true_continuation = []
            while mp_queue.qsize() > prefetch_min:
                time.sleep(0.05)

    if len(context_tokens) > 0:
        mp_queue.put((i - len(context_tokens) + 1, context_tokens, true_continuation))
        context_tokens = []
        true_continuation = []
    
    mp_queue.put((None, None, None))
    
    



def score(model, context_tokens, true_continuation):
    with torch.no_grad():
        context_tokens = torch.tensor(context_tokens, device = 'cuda')
        true_continuation = torch.tensor(true_continuation, device = 'cuda')

        generations = model.generate(context_tokens, temperature = 0.0, top_k = 0, top_p = 0, max_length = 64, min_length = 64)


        accuracies = (true_continuation == generations[:,32:64]).float().mean(axis=-1)
        return accuracies.cpu()

def main():
    # Extracting environment variables and miscellaneous initializations
    BATCH_SIZE = 512
    LOG_INTERVAL = 100 # Log every nth batch evals

    # Distributed variables
    RANK = int(os.environ['SLURM_PROCID'])
    LOCAL_RANK = int(os.environ['SLURM_LOCALID'])
    NUM_PROCS = int(os.environ['SLURM_NPROCS'])

    # Eval configuration variables
    MODEL = os.environ['MODEL']
    CHECKPOINT = int(os.environ['CHECKPOINT'])

    # Distributed initializations
    os.environ['MASTER_ADDR'] = os.environ['SLURM_LAUNCH_NODE_IPADDR']
    os.environ['MASTER_PORT'] = '12128'
    logging.basicConfig(format = f'rank-{RANK}:' + '%(levelname)s:%(message)s', level = logging.INFO)
    logging.info(f"Initializing torch distributed")

    # Initialize torch distributed
    dist.init_process_group(
        "nccl",
        world_size = NUM_PROCS,
        rank = RANK,
        timeout = datetime.timedelta(hours = 3)
    )
    store = dist.TCPStore(os.environ['MASTER_ADDR'], 12125, world_size = NUM_PROCS, is_master = RANK == 0, timeout = datetime.timedelta(hours=3))
    torch.cuda.set_device(LOCAL_RANK)

    if RANK == 0:
        store.set("currrank", "8")

    dist.barrier()
    while int(store.get("currrank").decode()) < RANK:
        time.sleep(1)

    # Model initialization
    transformer_utils.logging.set_verbosity_error()

    # Calculate start and end sequence indicies
    total_num_sequences = CHECKPOINT*1024
    num_sequences_per_proc = total_num_sequences//NUM_PROCS
    start_idx = num_sequences_per_proc*RANK
    end_idx = num_sequences_per_proc*(RANK+1) - 1
    if RANK == (NUM_PROCS -1):
        end_idx = total_num_sequences - 1

    # Dataset Initialization
    mp_queue = mp.Queue()
    ds_process = mp.Process(target = generate_dataset, args=(BATCH_SIZE, start_idx, end_idx, mp_queue))
    ds_process.start()

    currrank = int(store.get("currrank"))
    currrank = max(currrank, RANK + 8)
    store.set("currrank", str(currrank))

    # Model initialization
    model = AutoModelForCausalLM.from_pretrained(
        f"EleutherAI/pythia-{MODEL}",
        use_cache=False,
        revision = f'step{CHECKPOINT}',
        cache_dir=f"/fsx/orz/models/"
    ).half().eval().cuda()
    dist.barrier()

    # Run generations
    memorization_evals = []
    iters = 0
    while(True):
        try:
            t = time.time()
            idx, context, true_continuation = mp_queue.get()
            if idx is None:
                mp_queue.close()
                break

            idx = idx
            logging.info(f"Loading data took {time.time() - t:.3}s")
            t = time.time()
            accuracies = score(model, context, true_continuation)

            for acc in accuracies:
                memorization_evals.append(f'{idx},{acc}')
                idx += 1
            logging.info(f"Generation uptil {idx} took {time.time() - t:.3}s")
            dist.barrier()
            iters += 1
        except StopIteration:
            break
    
    ds_process.join()
    # Uploading evals to s3
    s3 = boto3.client('s3')
    s3.put_object(
        Body = '\n'.join(memorization_evals).encode(),
        Bucket = 's-eai-neox',
        Key = f'memorization-evals/memorization_{MODEL}_{CHECKPOINT}/rank-{RANK}.csv'
    )

    return
if __name__ == '__main__':
    mp.set_start_method('spawn')
    try:
        main()
    except RuntimeError as err:
        import requests
        import datetime
        import socket
        ts = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')+'UTC'
        resp = requests.get('http://169.254.169.254/latest/meta-data/instance-id')
        print(f'ERROR for {socket.gethostname()} at {ts} on {resp.text} device: {type(err).__name__}: {err}', flush=True)
        raise err