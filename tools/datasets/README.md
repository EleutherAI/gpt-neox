# Data Scripts

* `preprocess_data.py` takes a raw dataset, splits it up, tokenizes it, and saves it as numpy files that can be memmapped and used efficiently by the training code.
* `preprocess_data_with_mask.py` does the same but also creates `label` tensors if the dataset has labels.
* `multinode_prepare_data.sh` does the same but distributed over multiple nodes.
* `corpora.py` has information for common datasets.
