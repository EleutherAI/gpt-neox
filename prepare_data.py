from tools.corpora import prepare_dataset

if __name__ == "__main__":
    # with default tokenizer:
    prepare_dataset(dataset_name='enron', tokenizer_type='GPT2BPETokenizer', data_dir=None, vocab_file=None, merge_file=None)
    # with HF's GPT2TokenizerFast:
    # prepare_dataset(dataset_name='enron', tokenizer_type='HFGPT2Tokenizer', data_dir=None, vocab_file=None, merge_file=None)
    # with custom HF tokenizer:
    # prepare_dataset(dataset_name='enron', tokenizer_type='HFTokenizer', data_dir=None, vocab_file='data/tokenizer_vocab_file.json', merge_file=None)
    # with CharLevelTokenizer:
    # prepare_dataset(dataset_name='enwik8', tokenizer_type='CharLevelTokenizer', data_dir=None, vocab_file=None,
    #                     merge_file=None, num_workers=1)