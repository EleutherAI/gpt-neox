import copy
import pandas as pd

"""
Hardcoded variables that provide a list of the datasets used by T0. 

**Make sure Promptsource version is v0.1.0 for comparison with T0!!**
"""
# SAMPLE_TEMPLATE_LIST = {
#     # (dataset name, subset name): [template_name1, template_name2, ...]
#     ("super_glue", "copa"): ["exercise", "i_am_hesitating", "\u2026which may be caused by"],
#     ("super_glue", "rte"): [], # use all available templates if none given.
# }

T0_TRAIN_DATASETS = [    
]

with open('tools/p3.csv', newline='\n') as csvfile:
    df = pd.read_csv(csvfile)

for i, row in df.iterrows():
    if row["do_train"] == "BASE" and row["HF_name"] != "kilt_tasks":
        if pd.isnull(row["subset"]):
            subset = None
        else:
            subset = row["subset"]
        T0_TRAIN_DATASETS.append({(row["HF_name"], subset): []})

print(T0_TRAIN_DATASETS)

T0_PLUS_TRAIN_DATASETS = [
    *T0_TRAIN_DATASETS,
]


T0_PLUSPLUS_TRAIN_DATASETS = [
    *T0_PLUS_TRAIN_DATASETS,
]


T0_EVAL = [
]