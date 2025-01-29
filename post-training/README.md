# Post-Training

Examples for running post-training with ultrafeedback data for SFT/DPO/RM training.

For [REINFORCE](https://arxiv.org/abs/2402.14740) style training, see [Online Training](OnlineTraining.MD).

```bash
python tools/ckpts/convert_hf_llama_to_neox.py --tp 4 --model meta-llama/Meta-Llama-3-8B-Instruct --model_path checkpoints/neox_converted/llama3-8b-instruct
```

## Data generation
First, grab the jsonl file...

```bash
python post-training/llama_data.py
```
## DPO data
```bash
python tools/datasets/preprocess_data_with_chat_template.py --input data/pairwise/llama3_dpo_train_filtered.jsonl --output-prefix data/pairwise/llama3_dpo_train --tokenizer-path checkpoints/neox_converted/llama3-8b-instruct/tokenizer --jsonl-keys rejected --only-last
python tools/datasets/preprocess_data_with_chat_template.py --input data/pairwise/llama3_dpo_test_filtered.jsonl --output-prefix data/pairwise/llama3_dpo_test --tokenizer-path checkpoints/neox_converted/llama3-8b-instruct/tokenizer --jsonl-keys rejected --only-last
python tools/datasets/preprocess_data_with_chat_template.py --input data/pairwise/llama3_dpo_train_filtered.jsonl --output-prefix data/pairwise/llama3_dpo_val --tokenizer-path checkpoints/neox_converted/llama3-8b-instruct/tokenizer --jsonl-keys rejected --only-last
python tools/datasets/preprocess_data_with_chat_template.py --input data/pairwise/llama3_dpo_train_filtered.jsonl --output-prefix data/pairwise/llama3_dpo_train --tokenizer-path checkpoints/neox_converted/llama3-8b-instruct/tokenizer --jsonl-keys chosen --only-last
python tools/datasets/preprocess_data_with_chat_template.py --input data/pairwise/llama3_dpo_test_filtered.jsonl --output-prefix data/pairwise/llama3_dpo_test --tokenizer-path checkpoints/neox_converted/llama3-8b-instruct/tokenizer --jsonl-keys chosen --only-last
python tools/datasets/preprocess_data_with_chat_template.py --input data/pairwise/llama3_dpo_train_filtered.jsonl --output-prefix data/pairwise/llama3_dpo_val --tokenizer-path checkpoints/neox_converted/llama3-8b-instruct/tokenizer --jsonl-keys chosen --only-last
```

## RM data
```bash
python tools/datasets/preprocess_data_with_chat_template.py --input data/pairwise/llama3_dpo_train_filtered.jsonl --output-prefix data/pairwise/llama3_rm_train --tokenizer-path checkpoints/neox_converted/llama3-8b-instruct/tokenizer --jsonl-keys rejected --for-rm
python tools/datasets/preprocess_data_with_chat_template.py --input data/pairwise/llama3_dpo_test_filtered.jsonl --output-prefix data/pairwise/llama3_rm_test --tokenizer-path checkpoints/neox_converted/llama3-8b-instruct/tokenizer --jsonl-keys rejected --for-rm
python tools/datasets/preprocess_data_with_chat_template.py --input data/pairwise/llama3_dpo_train_filtered.jsonl --output-prefix data/pairwise/llama3_rm_val --tokenizer-path checkpoints/neox_converted/llama3-8b-instruct/tokenizer --jsonl-keys rejected --for-rm
python tools/datasets/preprocess_data_with_chat_template.py --input data/pairwise/llama3_dpo_train_filtered.jsonl --output-prefix data/pairwise/llama3_rm_train --tokenizer-path checkpoints/neox_converted/llama3-8b-instruct/tokenizer --jsonl-keys chosen --for-rm
python tools/datasets/preprocess_data_with_chat_template.py --input data/pairwise/llama3_dpo_test_filtered.jsonl --output-prefix data/pairwise/llama3_rm_test --tokenizer-path checkpoints/neox_converted/llama3-8b-instruct/tokenizer --jsonl-keys chosen --for-rm
python tools/datasets/preprocess_data_with_chat_template.py --input data/pairwise/llama3_dpo_train_filtered.jsonl --output-prefix data/pairwise/llama3_rm_val --tokenizer-path checkpoints/neox_converted/llama3-8b-instruct/tokenizer --jsonl-keys chosen --for-rm
```

## SFT data
```bash
python tools/datasets/preprocess_data_with_chat_template.py --input data/sft/llama3_sft_train_filtered.jsonl --output-prefix data/sft/llama3_train --tokenizer-path checkpoints/neox_converted/llama3-8b-instruct/tokenizer --jsonl-keys messages
python tools/datasets/preprocess_data_with_chat_template.py --input data/sft/llama3_sft_test_filtered.jsonl --output-prefix data/sft/llama3_test --tokenizer-path checkpoints/neox_converted/llama3-8b-instruct/tokenizer --jsonl-keys messages
python tools/datasets/preprocess_data_with_chat_template.py --input data/sft/llama3_sft_train_filtered.jsonl --output-prefix data/sft/llama3_val --tokenizer-path checkpoints/neox_converted/llama3-8b-instruct/tokenizer --jsonl-keys messages
```

## KTO data
```bash
python tools/datasets/preprocess_data_with_chat_template.py --input data/kto/llama3_sft_train_filtered.jsonl --output-prefix data/kto/llama3_train --tokenizer-path checkpoints/neox_converted/llama3-8b-instruct/tokenizer --jsonl-keys messages --reward-key reward
python tools/datasets/preprocess_data_with_chat_template.py --input data/kto/llama3_sft_test_filtered.jsonl --output-prefix data/kto/llama3_test --tokenizer-path checkpoints/neox_converted/llama3-8b-instruct/tokenizer --jsonl-keys messages --reward-key reward
python tools/datasets/preprocess_data_with_chat_template.py --input data/kto/llama3_sft_train_filtered.jsonl --output-prefix data/kto/llama3_val --tokenizer-path checkpoints/neox_converted/llama3-8b-instruct/tokenizer --jsonl-keys messages --reward-key reward
```


## Converting back to hf
```bash
# RM
python tools/ckpts/convert_neox_to_hf.py --input_dir eleuther-neox/checkpoints/rm/llama3/llama3-8b-instruct/global_step100 --output_dir checkpoints/rm/llama3_hf --config_file checkpoints/rm/llama3/llama3-8b-instruct/global_step100/configs/llama3-8b-rm.yml --precision bf16 --vocab-is-hf-tokenizer --architecture llama --pad-token-id 128002

# SFT/DPO
python tools/ckpts/convert_neox_to_hf.py --input_dir eleuther-neox/checkpoints/<dpo/sft>/llama3/llama3-8b-instruct/global_step100 --output_dir checkpoints/<dpo/sft>/llama3_hf --config_file checkpoints/<dpo/sft>/llama3/llama3-8b-instruct/global_step100/configs/llama3-8b-rm.yml --precision bf16 --vocab-is-hf-tokenizer --architecture llama
```
