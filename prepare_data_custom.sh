python tools/preprocess_data.py \
            --input /cognitive_comp/wuziwei/test_data/gpt-neox/test_data.jsonl \
            --jsonl-keys text \
            --output-prefix ./data/test_data \
            --vocab /cognitive_comp/common_data/BPETokenizer-Mix-NEO-pre \
            --dataset-impl mmap \
            --tokenizer-type HFGPTNeoXTokenizerFast \
            --workers 2 \
            --log-data \
            --append-eod