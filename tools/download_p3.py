import json
import sys

from typing import List

from promptsource.templates import DatasetTemplates
from datasets import load_dataset

from p3_configs import T0_TRAIN_DATASETS

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

    # TODO(Hailey): should we replicate the limiting ds examples for t0 exactly?
    # this seems like it does weird stuff.... (from t-zero repo)
    # if train_size=499k, then cap = 499k but if == 500k and 12 templates, then cap = ~41k.

    # if train_size > MAX_EXAMPLES_PER_DATASET:
    #     cap = MAX_EXAMPLES_PER_DATASET // num_templates
    # else:
    #     cap = train_size

    ds = load_dataset(dataset_name, name=subset_name, cache_dir=cache_dir)

    ds = ds[split]

    if limit is not None:
        limit = min(limit, len(ds))
        ds = ds.select(range(limit))

    templates = DatasetTemplates(dataset_name, subset_name=subset_name)

    metadata = f"{dataset_name}_{subset_name}_{split}" if subset_name else f"{dataset_name}_{split}"

    with open(f'./data/p3_raw/p3_{sys.argv[1]}.jsonl', mode='w') as f:
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
                # do strip() on targets because Promptsource always has leading whitespace in returned targets
                f.write(json.dumps({
                    "inputs": inputs, 
                    "targets": targets[0].strip(), }) + "\n")


# def apply_to_tf_dataset()
# TODO(Hailey): implement this function to apply FLAN prompts to a FLAN TFDS dataset, or HF dataset?
# TODO(Hailey): check equivalence between TFDS and HF datasets for the datasets we care about. '
# Being able to just work on one dataset lib/format would be fantastic (HF datasets)


if __name__ == '__main__':

    # define your mixtures in p3_configs.py .
    # change these assignments to use custom dataset mixtures + sets of prompts.
    all_tasks = SAMPLE_TEMPLATE_LIST.keys()
    all_prompts = SAMPLE_TEMPLATE_LIST

    for task in all_tasks:
        ds_name, subset = task
        apply_to_hf_dataset(ds_name, all_prompts[task], subset_name=subset, limit=3000, split=sys.argv[1])


