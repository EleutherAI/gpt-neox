import json

from typing import List

from promptsource.templates import DatasetTemplates
from datasets import load_dataset

from p3_configs import T0_TRAIN_TEMPLATES

# stores all names of promptsource templates to use as lists of strings
# indexed by (dataset name, subset name) tuples
SAMPLE_TEMPLATE_LIST = {
    # (dataset name, subset name): [template_name1, template_name2, ...]
    ("super_glue", "copa"): ["exercise", "i_am_hesitating", "\u2026which may be caused by"],
    ("super_glue", "rte"): [], # use all available templates if none given.
}


def apply_to_hf_dataset(
    dataset_name: str,
    prompts: List[str],
    subset_name: str = None, 
    limit: int = 500_000, # t0 does this, see https://github.com/bigscience-workshop/t-zero/blob/master/t0/seqio_tasks/tasks.py
    split: str = "train",
    cache_dir = None,
    ):
    """
    Applies all given prompts to the first `limit` examples of a dataset split.
    """
    ds = load_dataset(dataset_name, name=subset_name, cache_dir=cache_dir)

    ds = ds[split]

    if limit is not None:
        limit = min(limit, len(ds))
        ds = ds.select(range(limit))

    templates = DatasetTemplates(dataset_name, subset_name=subset_name)

    metadata = f"{dataset_name}_{subset_name}_{split}" if subset_name else f"{dataset_name}_{split}"

    with open(f'./data/p3_raw/p3_train.jsonl', mode='w') as f:
        if len(prompts) == 0:
            prompts = templates.all_template_names
        for prompt in prompts:
            template = templates[prompt]

            for example in ds:
                try:
                    inputs, targets = template.apply(example)
                except:
                    # skip example if we cannot unpack
                    continue

                # TODO: how to deal with multiple-target examples? rn we take the first one always
                f.write(json.dumps({
                    "inputs": inputs, 
                    "targets": targets[0].strip(), }) + "\n")

if __name__ == '__main__':

    # change these assignments to use custom dataset mixtures + sets of prompts
    all_tasks = SAMPLE_TEMPLATE_LIST.keys()
    all_prompts = SAMPLE_TEMPLATE_LIST

    for task in all_tasks:
        ds_name, subset = task
        apply_to_hf_dataset(ds_name, all_prompts[task], subset_name=subset, limit=3000, split="train")


