python tools/preprocess_data.py \
            --input /cognitive_comp/wuziwei/test_data/gpt-neox/test_data.jsonl \
            --jsonl-keys text \
            --output-prefix ./data/mydataset \
            --vocab /cognitive_comp/wuziwei/codes/gpt-neox/custom_config/10B/20B_tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --workers 1 \
            --append-eod