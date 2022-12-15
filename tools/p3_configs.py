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

T0_TRAIN_DATASETS = {
    ('glue', 'mrpc'): [], 
    ('glue', 'qqp'): [], 
    ('paws', 'labeled_final'): [], 
    ('kilt_tasks', 'hotpotqa'): [], 
    ('wiki_qa', None): [], 
    ('adversarial_qa', 'dbidaf'): [], 
    ('adversarial_qa', 'dbert'): [], 
    ('adversarial_qa', 'droberta'): [], 
    ('duorc', 'SelfRC'): [], 
    ('duorc', 'ParaphraseRC'): [], 
    ('ropes', None): [], 
    ('quoref', None): [], 
    ('cos_e', 'v1.11'): [], 
    ('cosmos_qa', None): [], 
    ('dream', None): [], 
    ('qasc', None): [], 
    ('quail', None): [], 
    ('quarel', None): [], 
    ('quartz', None): [], 
    ('sciq', None): [], 
    ('social_i_qa', None): [], 
    ('wiki_hop', 'original'): [], 
    ('wiqa', None): [], 
    ('amazon_polarity', None): [], 
    ('app_reviews', None): [], 
    ('imdb', None): [], 
    ('rotten_tomatoes', None): [], 
    ('yelp_review_full', None): [], 
    ('common_gen', None): [], 
    ('wiki_bio', None): [], 
    ('cnn_dailymail', '3.0.0'): [], 
    ('gigaword', None): [], 
    ('multi_news', None): [], 
    ('samsum', None): [], 
    ('xsum', None): [], 
    ('ag_news', None): [], 
    ('dbpedia_14', None): [], 
    ('trec', None): [],
}

T0_PLUS_TRAIN_DATASETS = [
    *T0_TRAIN_DATASETS,
]


T0_PLUSPLUS_TRAIN_DATASETS = [
    *T0_PLUS_TRAIN_DATASETS,
]


T0_EVAL = [
]


if __name__ == "__main__":
    with open('tools/p3.csv', newline='\n') as csvfile:
        df = pd.read_csv(csvfile)
    for i, row in df.iterrows():
        if row["do_train"] == "BASE" and row["HF_name"] != "kilt_tasks":
            if pd.isnull(row["subset"]):
                subset = None
            else:
                subset = row["subset"]
        T0_TRAIN_DATASETS.append({(row["HF_name"], subset): []}) 
