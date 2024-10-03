"""
https://github.com/huggingface/alignment-handbook/blob/main/scripts/run_dpo.py
adapted to just grab the dataset
"""
import os
from alignment import (
    DataArguments,
    DPOConfig,
    H4ArgumentParser,
    ModelArguments,
    apply_chat_template,
    decontaminate_humaneval,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

import jsonlines

###############
# Load datasets
###############
raw_datasets = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
raw_datasets = DatasetDict(
    {
        "train": raw_datasets["train_prefs"],
        "test": raw_datasets["test_prefs"],
    }
)
column_names = list(raw_datasets["train"].features)

#####################################
# Load tokenizer and process datasets
#####################################
truncation_side = (
    "left"  # Truncate from left to ensure we don't lose labels in final turn
)
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")

#####################
# Apply chat template
#####################
raw_datasets = raw_datasets.map(
    apply_chat_template,
    fn_kwargs={
        "tokenizer": tokenizer,
        "task": "dpo",
        "auto_insert_empty_system_msg": True,
    },
    desc="Formatting comparisons with prompt template",
)

##########################
# Decontaminate benchmarks
##########################
num_raw_train_samples = len(raw_datasets["train"])
raw_datasets = raw_datasets.filter(
    decontaminate_humaneval,
    fn_kwargs={"text_column": "text_chosen"},
    batched=True,
    batch_size=10_000,
    num_proc=1,
    desc="Decontaminating HumanEval samples",
)
num_filtered_train_samples = num_raw_train_samples - len(raw_datasets["train"])
print(
    f"Decontaminated {num_filtered_train_samples} ({num_filtered_train_samples / num_raw_train_samples * 100:.2f}%) samples from the training set."
)
###############
# Length filter
###############
# Since the alignment handbook recipes call for a max token limit of 1024...
num_filtered_train_samples = len(raw_datasets["train"])


def length_filter(example):
    return (len(tokenizer.apply_chat_template(example["chosen"])) < 1024) and (
        len(tokenizer.apply_chat_template(example["rejected"])) < 1024
    )


num_length_filtered_train_samples = num_filtered_train_samples - len(
    raw_datasets["train"]
)
print(
    f"Length Filtered {num_length_filtered_train_samples} ({num_length_filtered_train_samples / num_filtered_train_samples * 100:.2f}%) samples from the training set."
)
# get directory of the python script
dir_path = os.path.dirname(os.path.realpath(__file__))
for split in ["train", "test"]:
    with open(os.path.join(dir_path, f"dpo_{split}_filtered.jsonl"), "w") as f:
        writer = jsonlines.Writer(f)
        for item in raw_datasets[split]:
            # add empty system messages
            item["chosen"] = [{"role": "system", "content": ""}] + item["chosen"]
            item["rejected"] = [{"role": "system", "content": ""}] + item["rejected"]
            writer.write(item)
