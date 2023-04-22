import argparse
from time import time
import logging
from datasets import load_from_disk
from transformers import GPT2TokenizerFast
from tokenizers import (
    normalizers,
    decoders,
    Tokenizer,
)
from tokenizers.pre_tokenizers import (
    Punctuation,
    Digits,
    ByteLevel,
    UnicodeScripts,
    Sequence,
)
import emoji
from tokenizers.models import BPE, Unigram
from tokenizers.trainers import BpeTrainer, UnigramTrainer
from utils import load_dataset, batch_iterator


logger = logging.getLogger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="bpe",
        choices=["bpe", "unigram"],
        help="tokenizer model",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="dropout rate for BPE"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=102400, help="vocab size for tokenizer"
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="tokenizer")
    parser.add_argument(
        "--normalizer",
        type=str,
        default="NFKC",
        choices=["NFKC", "NFC"],
        help="unicode normalizer",
    )
    parser.add_argument(
        "--continuing_subword_prefix",
        type=str,
        default="",
        help="valid in BPE. empty string reverts to disable. ## is common symbol. ",
    )
    parser.add_argument(
        "--cache_capacity",
        type=int,
        default=10000,
        help="cache_capacity in BPE.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="number of processes to use. num_proc=0 uses all cores",
    )
    parser.add_argument(
        "--buffer_tokens",
        type=int,
        default=100,
        help="number of buffer tokens to pad before tokenizer initialization",
    )
    args = parser.parse_args()
    return args


def main(args):
    if args.num_proc == 0:
        import multiprocessing

        num_proc = multiprocessing.cpu_count()
    else:
        num_proc = args.num_proc

    if ".json" in args.data_path:
        dataset = load_dataset(args.data_path, num_proc=num_proc)
    else:
        dataset = load_from_disk(args.data_path)

    # tokenizer arguments
    SPECIAL_TOKENS = [
        "<s>",
        "</s>",
        "<usr>",
        "<pad>",
        "<sys>",
        "<unk>",
        "<mask>",
        "<d>",
        "</d>",
    ]
    with open("facial_expression.txt") as f:
        facial_expression = [line.strip() for line in f.readlines()]
    emoji_unicode_face = list(
        set(
            [
                emoji.EMOJI_UNICODE["en"][i][0]
                for i in emoji.EMOJI_UNICODE["en"].keys()
                if "face" in i
            ]
        )
    )
    # start with buffer tokens
    buffer_tokens = [f"<unused{i}>" for i in range(args.buffer_tokens)]
    # calculate whitespace tokens
    whitespace = " "
    whitespace_count = 2
    whitespace_list = [
        whitespace * (2**count) for count in range(whitespace_count, 0, -1)
    ]
    # construct added_tokens
    added_tokens = (
        SPECIAL_TOKENS
        + facial_expression
        + emoji_unicode_face
        + buffer_tokens
        + whitespace_list
    )

    # tokenizer normalizer
    if args.model.lower() == "nfc":
        normalizer = normalizers.NFC()
    elif args.model.lower() == "nfkc":
        normalizer = normalizers.NFKC()
    # common pretokenizer
    pre_tokenizer = Sequence(
        [
            UnicodeScripts(),  # split on different unicode range
            Punctuation(
                behavior="isolated",
            ),
            Digits(individual_digits=True),
            ByteLevel(add_prefix_space=False, use_regex=True),
        ]
    )
    # common decoder
    decoder = decoders.ByteLevel(add_prefix_space=False, use_regex=True)
    if args.model.lower() == "bpe":
        tokenizer = Tokenizer(BPE(dropout=args.dropout))
        trainer = BpeTrainer(
            vocab_size=args.vocab_size,
            special_tokens=added_tokens,
            initial_alphabet=ByteLevel.alphabet(),
        )
    else:
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(
            vocab_size=args.vocab_size,
            special_tokens=added_tokens,
            initial_alphabet=ByteLevel.alphabet(),
        )
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.decoder = decoder

    start = time()
    tokenizer.train_from_iterator(
        batch_iterator(dataset), trainer=trainer, length=len(dataset)
    )
    end = time()
    print(f"time collapsed: {(end - start) / 60:.2f}m")
    tokenizer_wrapper = GPT2TokenizerFast(
        tokenizer_object=tokenizer,
        vocab_size=args.vocab_size,
        additional_special_tokens=SPECIAL_TOKENS,
        bos_token=SPECIAL_TOKENS[0],
        eos_token=SPECIAL_TOKENS[0],
        unk_token=SPECIAL_TOKENS[0],
    )
    text = "아!@ 진짜 억울해죽것네'''아니english123배고파씌 korea으😣악😣😳😣'''"
    print(f"text: {text}")
    tokens = tokenizer_wrapper.tokenize(text)
    input_ids = tokenizer_wrapper(text)["input_ids"]
    print(f"tokens: {tokens}")
    print(f"decode: {tokenizer_wrapper.decode(input_ids)}")
    tokenizer_wrapper.save_pretrained(f"{args.save_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
