# Data Scripts

## `preprocess_data.py`
Takes a raw dataset, splits it up, tokenizes it, and saves it as numpy files that can be memmapped and used efficiently by the training code.

```
usage: preprocess_data.py [-h] --input INPUT [--jsonl-keys JSONL_KEYS [JSONL_KEYS ...]] [--num-docs NUM_DOCS]
                          --tokenizer-type
                          {HFGPT2Tokenizer,HFTokenizer,GPT2BPETokenizer,CharLevelTokenizer,TiktokenTokenizer,SPMTokenizer}
                          [--vocab-file VOCAB_FILE] [--merge-file MERGE_FILE] [--append-eod] [--ftfy] --output-prefix
                          OUTPUT_PREFIX [--dataset-impl {lazy,cached,mmap}] [--workers WORKERS]
                          [--log-interval LOG_INTERVAL]

options:
  -h, --help            show this help message and exit

input data:
  --input INPUT         Path to input jsonl files or lmd archive(s) - if using multiple archives, put them in a comma
                        separated list
  --jsonl-keys JSONL_KEYS [JSONL_KEYS ...]
                        space separate listed of keys to extract from jsonl. Defa
  --num-docs NUM_DOCS   Optional: Number of documents in the input data (if known) for an accurate progress bar.

tokenizer:
  --tokenizer-type {HFGPT2Tokenizer,HFTokenizer,GPT2BPETokenizer,CharLevelTokenizer,TiktokenTokenizer,SPMTokenizer}
                        What type of tokenizer to use.
  --vocab-file VOCAB_FILE
                        Path to the vocab file
  --merge-file MERGE_FILE
                        Path to the BPE merge file (if necessary).
  --append-eod          Append an <eod> token to the end of a document.
  --ftfy                Use ftfy to clean text

output data:
  --output-prefix OUTPUT_PREFIX
                        Path to binary output file without suffix
  --dataset-impl {lazy,cached,mmap}
                        Dataset implementation to use. Default: mmap

runtime:
  --workers WORKERS     Number of worker processes to launch
  --log-interval LOG_INTERVAL
                        Interval between progress updates
```
## `preprocess_data_with_mask.py`
Does the same but also creates `label` tensors if the dataset has labels.

N.B. If using this, you  **must** specify your data when training/finetuning with the following configs
```json
"train_data_paths": ["train_documents"],
"test_data_paths": ["test_documents"],
"valid_data_paths": ["test_documents"],
"label_data_paths": ["label_documents"]
```

the `"data_path"` option will not work with `"label_data_paths"`.


```
usage: preprocess_data_with_mask.py [-h] --input INPUT [--jsonl-keys JSONL_KEYS [JSONL_KEYS ...]]
                                    [--mask-before-token MASK_BEFORE_TOKEN] [--num-docs NUM_DOCS] --tokenizer-type
                                    {HFGPT2Tokenizer,HFTokenizer,GPT2BPETokenizer,CharLevelTokenizer}
                                    [--vocab-file VOCAB_FILE] [--merge-file MERGE_FILE] [--append-eod] [--ftfy]
                                    --output-prefix OUTPUT_PREFIX [--dataset-impl {lazy,cached,mmap}]
                                    [--workers WORKERS] [--log-interval LOG_INTERVAL]

options:
  -h, --help            show this help message and exit

input data:
  --input INPUT         Path to input jsonl files or lmd archive(s) - if using multiple archives, put them in a comma
                        separated list
  --jsonl-keys JSONL_KEYS [JSONL_KEYS ...]
                        space separate listed of keys to extract from jsonl. Defa
  --mask-before-token MASK_BEFORE_TOKEN
                        apply loss masks before certain token(s). If multi-token pattern, separate by commas without
                        space, e.g. --mask-before-token 0,1,1270 to use the token pattern [0,1,1270].
  --num-docs NUM_DOCS   Optional: Number of documents in the input data (if known) for an accurate progress bar.

tokenizer:
  --tokenizer-type {HFGPT2Tokenizer,HFTokenizer,GPT2BPETokenizer,CharLevelTokenizer}
                        What type of tokenizer to use.
  --vocab-file VOCAB_FILE
                        Path to the vocab file
  --merge-file MERGE_FILE
                        Path to the BPE merge file (if necessary).
  --append-eod          Append an <eod> token to the end of a document.
  --ftfy                Use ftfy to clean text

output data:
  --output-prefix OUTPUT_PREFIX
                        Path to binary output file without suffix
  --dataset-impl {lazy,cached,mmap}
                        Dataset implementation to use. Default: mmap

runtime:
  --workers WORKERS     Number of worker processes to launch
  --log-interval LOG_INTERVAL
                        Interval between progress updates
```
## `multinode_prepare_data.sh`
Does the same but distributed over multiple nodes.

```
# USAGE:
# This script allows you to prepare your dataset using multiple nodes by chunking the individual files and distributed the chunks
# over the processes.
# This bash script takes a single text file as input argument.
# The text file contains a valid filepath in each line, leading to a jsonl-file.
# Furthermore an environment variable for the rank and the world size needs to be set.
# These default to the SLURM and OMPI variables in this order of priority, but they can be set manually as well
# using the variables $RANK and $WORLD_SIZE, which will overwrite the cluster-specific variables.
# You can also add all arguments of the prepare_data.py script to this script and it will simply pass them through.
```


## `corpora.py`
Has information for common datasets. Primarily meant for use in top-level `prepare_data.py` script.
