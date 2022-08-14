python tools/preprocess_data.py \
    --input '/fsx/multi-lingual-6b/gpt-neox/data/*/part-00000-*.jsonl' \
    --input_with_pattern \
    --output-prefix ./processed/multi_ko \
    --vocab tokenizer/MBBPE/tokenizer.json \
    --dataset-impl mmap \
    --tokenizer-type HFTokenizer \
    --append-eod \
    --workers 16 \
    --chunksize 100
