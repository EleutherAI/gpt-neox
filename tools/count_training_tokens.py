import argparse
import time
import logging
from glob import glob
from functools import reduce
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd


def main():
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"fsx/multi-lingual-6b/gpt-neox/logs/{current_time}.log"),
            logging.StreamHandler()
        ]
    )
    tokenizer = AutoTokenizer.from_pretrained("multilingual-transfer/tokenizer/MBBPE")
    data_files = glob("/fsx/multi-lingual-6b/gpt-neox/data/*/part-00000")
    dataset = load_dataset("json", data_files=data_files, split="train", streaming=True)

    def _sum(x, y):
        return x + y

    def encode(examples):
        encoded = tokenizer(examples['text'])
        num_examples = len(encoded['input_ids'])
        num_tokens = reduce(_sum, list(map(lambda example: len(example), encoded['input_ids'])))
        return {'num_tokens': [num_tokens], 'num_examples': [num_examples]}
    
    dataset = dataset.map(encode, batched=True, batch_size=10000, remove_columns=["text"])
    
    counter = 0
    num_tokens = 0
    start = pd.Timestamp.now()
    for example in dataset:
        num_tokens += example['num_tokens']
        counter += example['num_examples']
        if counter % 100000 == 0:
            logging.info(f'{counter} examples are processed (cumulative number of tokens: {num_tokens}).')

    end = pd.Timestamp.now()
    logging.info(f"collapsed time: {end - start}")
    logging.info(f"total number of examples: {counter}")
    logging.info(f"total number of tokens: {num_tokens}")
    logging.info(f"total number of tokens including special tokens: {num_tokens + counter}")

if __name__ == "__main__":
    main()