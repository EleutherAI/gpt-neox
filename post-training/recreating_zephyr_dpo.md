# Initial setup

```bash
python tools/ckpts/convert_hf_llama_to_neox.py --tp 2 --model HuggingFaceH4/mistral-7b-sft-beta --model_path checkpoints/neox_converted/zephyr-sft_tp2
```


# To generate data
First make a new environment... We want to keep the same data between runs so the easiest way is to create a new conda
environment and follow the steps below.
```
conda create -n handbook python=3.10 && conda activate handbook
git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
python -m pip install .
python -m pip install jsonlines
```

## DPO data
```bash
# from the gpt-neox repo
conda activate handbook
python post-training/dpo_data.py
conda deactivate
# activate your neox conda environment, or whatever you need to switch to the neox environment
mkdir data
mkdir data/pairwise
python tools/datasets/preprocess_data_with_chat_template.py --input post-training/dpo_train_filtered.jsonl --output-prefix data/pairwise/dpo_train --tokenizer-path checkpoints/neox_converted/zephyr-sft/tokenizer --jsonl-keys rejected --only-last
python tools/datasets/preprocess_data_with_chat_template.py --input post-training/dpo_test_filtered.jsonl --output-prefix data/pairwise/dpo_test --tokenizer-path checkpoints/neox_converted/zephyr-sft/tokenizer --jsonl-keys rejected --only-last
python tools/datasets/preprocess_data_with_chat_template.py --input post-training/dpo_train_filtered.jsonl --output-prefix data/pairwise/dpo_val --tokenizer-path checkpoints/neox_converted/zephyr-sft/tokenizer --jsonl-keys rejected --only-last
python tools/datasets/preprocess_data_with_chat_template.py --input post-training/dpo_train_filtered.jsonl --output-prefix data/pairwise/dpo_train --tokenizer-path checkpoints/neox_converted/zephyr-sft/tokenizer --jsonl-keys chosen --only-last
python tools/datasets/preprocess_data_with_chat_template.py --input post-training/dpo_test_filtered.jsonl --output-prefix data/pairwise/dpo_test --tokenizer-path checkpoints/neox_converted/zephyr-sft/tokenizer --jsonl-keys chosen --only-last
python tools/datasets/preprocess_data_with_chat_template.py --input post-training/dpo_train_filtered.jsonl --output-prefix data/pairwise/dpo_val --tokenizer-path checkpoints/neox_converted/zephyr-sft/tokenizer --jsonl-keys chosen --only-last
```

## Running
```bash
python deepy.py train.py post-training/configs/benchmarking/mistral-dpo.yml
```
