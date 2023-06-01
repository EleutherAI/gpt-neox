import datasets
from pathlib import Path

shard_size=500_000

def main(args): 
    """
    Downloads proof-pile-v2 from huggingface, without needing git lfs installed.
    """
    Path("data/train/proof-pile").mkdir(exist_ok=True, parent=True)
    Path("data/validation/proof-pile").mkdir(exist_ok=True, parent=True)
    Path("data/test/proof-pile").mkdir(exist_ok=True, parent=True)

    ds = ds["train"]

    for shard, left in enumerate(range(0, len(ds), shard_size)):
        ds.to_json(f"data/train/proof-pile/{shard}.jsonl")

    ds["validation"].to_json(f"data/validation/proof-pile/validation.jsonl")
    ds["test"].to_json(f"data/test/proof-pile/test.jsonl")


