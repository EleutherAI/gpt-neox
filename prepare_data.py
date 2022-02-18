from tools.corpora import prepare_dataset, DATA_DOWNLOADERS
import argparse

TOKENIZER_CHOICES = [
    "HFGPT2Tokenizer",
    "HFTokenizer",
    "GPT2BPETokenizer",
    "CharLevelTokenizer",
]
DATASET_CHOICES = [i for i in DATA_DOWNLOADERS.keys() if i != "pass"]


def get_args():
    parser = argparse.ArgumentParser(description="Download & preprocess neox datasets")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="enron",
        help="name of dataset to download.",
        choices=DATASET_CHOICES,
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        default="GPT2BPETokenizer",
        choices=TOKENIZER_CHOICES,
        help=f'Type of tokenizer to use - choose from {", ".join(TOKENIZER_CHOICES)}',
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        default=None,
        help=f"Directory to which to download datasets / tokenizer "
        f"files - defaults to ./data",
    )
    parser.add_argument(
        "-v", "--vocab-file", default=None, help=f"Tokenizer vocab file (if required)"
    )
    parser.add_argument(
        "-m", "--merge-file", default=None, help=f"Tokenizer merge file (if required)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    prepare_dataset(
        dataset_name=args.dataset,
        tokenizer_type=args.tokenizer,
        data_dir=args.data_dir,
        vocab_file=args.vocab_file,
        merge_file=args.merge_file,
    )
