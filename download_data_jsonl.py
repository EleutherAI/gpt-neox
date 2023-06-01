import datasets
from pathlib import Path

shard_size=500_000

def main(): 
    """
    Downloads proof-pile-v2 from huggingface, without needing git lfs installed.
    """
    Path("data/train/proof-pile").mkdir(exist_ok=True, parent=True)
    Path("data/validation/proof-pile").mkdir(exist_ok=True, parent=True)
    Path("data/test/proof-pile").mkdir(exist_ok=True, parent=True)

    ds = ds["train"]

    for shard, left in enumerate(range(0, len(ds), shard_size)):
        ds.select(left, min(left+shard_size, len(ds))).to_json(
                f"data/train/proof-pile/{str(shard).zfill(5)}.jsonl", 
                lines=True,
        )

    ds["validation"].to_json(f"data/validation/proof-pile/validation.jsonl", lines=True)
    ds["test"].to_json(f"data/test/proof-pile/test.jsonl", lines=True)

if __name__=="__main__": 
    main()
