import argparse
import os
import time
from datasets import load_from_disk, concatenate_datasets
from utils import load_dataset
from kss import split_morphemes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="data/processed")
    args = parser.parse_args()
    return args


def main(args):
    data_path = args.data_path
    data_path = data_path.replace("~", os.path.expanduser("~"))
    if ".json" in args.data_path:
        dataset = load_dataset(args.data_path)
    else:
        dataset = load_from_disk(args.data_path)

    num_proc = os.cpu_count()
    batch_size = 1000 * num_proc

    def process_morpheme(examples):
        result = {"text": []}
        try:
            processed = split_morphemes(
                examples["text"], drop_space=False, num_workers=num_proc
            )
            processed = [
                " ".join([token for token, _ in instance if token != " "])
                for instance in processed
            ]
        except:
            processed = []
            micro_batch_size = batch_size // 10
            for i in range(0, len(examples["text"]), micro_batch_size):
                try:
                    processed_batch = split_morphemes(
                        examples["text"][i : i + micro_batch_size],
                        drop_space=False,
                        num_workers=num_proc,
                    )
                    processed_batch = [
                        " ".join([token for token, _ in instance if token != " "])
                        for instance in processed
                    ]
                    processed += processed_batch
                except:
                    pass
        result["text"] = processed
        return result

    num_shards = ((len(dataset) - 1) // 1000000) + 1
    start = time.time()

    if num_shards > 1:
        dataset_shards = [
            dataset.shard(num_shards=num_shards, index=i) for i in range(num_shards)
        ]
        proc_dataset_shards = []
        for i, dataset in enumerate(dataset_shards):
            proc_dataset_shards.append(
                dataset.map(
                    process_morpheme,
                    batched=True,
                    batch_size=batch_size,
                    desc=f"Map {i+1}/{num_shards}",
                )
            )
        proc_dataset = concatenate_datasets(proc_dataset_shards)
    else:
        proc_dataset = dataset.map(
            process_morpheme, batched=True, batch_size=1000 * num_proc
        )
    end = time.time()
    print(f"time collapsed: {end - start:.2f}m")

    os.makedirs(args.save_dir, exist_ok=True)
    basename = os.path.basename(args.data_path)
    proc_dataset.save_to_disk(os.path.join(args.save_dir, basename))


if __name__ == "__main__":
    args = parse_args()
    main(args)
