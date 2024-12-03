import os

from datasets import load_dataset, DatasetDict

import jsonlines

###############
# Load datasets
###############
raw_datasets = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
# convert to just train and test, not necessary but it looks better
raw_datasets = DatasetDict(
    {
        "train": raw_datasets["train_prefs"],
        "test": raw_datasets["test_prefs"],
    }
)
os.makedirs(os.path.join("data", "pairwise"), exist_ok=True)
for split in ["train", "test"]:
    with open(
        os.path.join("data", "pairwise", f"llama3_dpo_{split}_filtered.jsonl"), "w"
    ) as f:
        writer = jsonlines.Writer(f)
        for item in raw_datasets[split]:
            item["chosen"] = item["chosen"]
            item["rejected"] = item["rejected"]
            writer.write(item)
os.makedirs(os.path.join("data", "sft"), exist_ok=True)
for split in ["train", "test"]:
    with open(
        os.path.join("data", "sft", f"llama3_sft_{split}_filtered.jsonl"), "w"
    ) as f:
        writer = jsonlines.Writer(f)
        for item in raw_datasets[split]:
            item["messages"] = item["chosen"]
            writer.write(item)
os.makedirs(os.path.join("data", "kto"), exist_ok=True)
for split in ["train", "test"]:
    with open(
        os.path.join("data", "kto", f"llama3_kto_{split}_filtered.jsonl"), "w"
    ) as f:
        writer = jsonlines.Writer(f)
        for item in raw_datasets[split]:
            item["messages"] = item["chosen"]
            item["reward"] = 1
            writer.write(item)
            item["messages"] = item["rejected"]
            item["reward"] = -1
            writer.write(item)
