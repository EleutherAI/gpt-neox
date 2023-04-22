import argparse
from time import time
import logging
from datasets import load_from_disk
from transformers import GPT2TokenizerFast
from tokenizers import (
    normalizers,
    decoders,
    Tokenizer,
    processors,
)
from tokenizers.pre_tokenizers import (
    Punctuation,
    Digits,
    ByteLevel,
    UnicodeScripts,
    Sequence,
    WhitespaceSplit,
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
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        # default="~/corpus/jsonl/the_stack_smol.jsonl",
        help="takes str to single file",
    )
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
        default=10000,  # this is default for tokenizers. TODO: ablations
        help="cache_capacity in BPE.",
    )
    parser.add_argument(
        "--buffer_tokens",
        type=int,
        default=100,
        help="number of tokens to pad BEFORE tokenizer initialization",
    )
    parser.add_argument(
        "--preserve_whitespace",
        type=str,
        default="yes",
        choices=["yes", "inference", "no"],
        help="choose whitespace preservation. \n \
            yes preserves during training and inference\n \
            inference removes during training but resumes at inference\n \
            no removes completely. this makes tokenizer non invertible(loses original)",
    )
    args, _ = parser.parse_known_args()
    return args


def main(args):
    if ".jsonl" in args.data_path:
        dataset = load_dataset(args.data_path)
    else:
        dataset = load_from_disk(args.data_path)  # returns dataset

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
    """with open("facial_expression.txt") as f:
        facial_expression = [line.strip() for line in f.readlines()]
    emoji_unicode_face = list(
        set(
            [
                emoji.EMOJI_UNICODE["en"][i][0]
                for i in emoji.EMOJI_UNICODE["en"].keys()
                if "face" in i
            ]
        )
    )"""
    # start with buffer tokens
    buffer_tokens = [f"<unused{i}>" for i in range(args.buffer_tokens)]
    # calculate whitespace tokens
    whitespace = " "
    whitespace_count = 2  # 4,2,1 whitespaces
    # necessary for invertibility
    if not args.preserve_whitespace == "yes":
        last_whitespace = -1
    else:
        last_whitespace = 0  # only
    whitespace_list = [
        whitespace * (2**count)
        for count in range(whitespace_count, last_whitespace, -1)
    ]
    # construct added_tokens
    added_tokens = (
        SPECIAL_TOKENS
        # + facial_expression
        # + emoji_unicode_face
        + buffer_tokens
        + whitespace_list
    )
    initial_alphabet = ByteLevel.alphabet()
    initial_alphabet.sort()

    # tokenizer normalizer
    if args.normalizer.lower() == "nfc":
        normalizer = normalizers.NFC()
    elif args.normalizer.lower() == "nfkc":
        normalizer = normalizers.NFKC()
    # common pretokenizer
    pre_tokenizer_list = [
        UnicodeScripts(),  # split on different unicode range
        Punctuation(
            behavior="isolated",
        ),
        Digits(individual_digits=True),
        ByteLevel(add_prefix_space=False, use_regex=True),
    ]
    # if yes, default to no whitespace handling
    if not args.preserve_whitespace == "yes":
        pre_tokenizer_list.insert(
            0,
            WhitespaceSplit(),
        )  # whitespace split should be in front
    pre_tokenizer = Sequence(pre_tokenizer_list)
    # common decoder
    decoder = decoders.ByteLevel(add_prefix_space=False, use_regex=True)
    if args.model.lower() == "bpe":
        tokenizer = Tokenizer(
            BPE(
                cache_capacity=args.cache_capacity,
                dropout=args.dropout,
                byte_fallback=False,
            )
        )
        trainer = BpeTrainer(
            vocab_size=args.vocab_size,
            special_tokens=added_tokens,
            initial_alphabet=initial_alphabet,
            continuing_subword_prefix=args.continuing_subword_prefix,
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
    print(f"time elapsed: {(end - start) / 60:.2f}m")
    # if preserve whitespace is set to "inference", remove Whitespace splitter
    if args.preserve_whitespace == "inference":
        for idx in range(len(pre_tokenizer_list)):
            if isinstance(pre_tokenizer_list[idx], WhitespaceSplit):
                pre_tokenizer_list.pop(idx)
                break
    tokenizer.pre_tokenizer = Sequence(pre_tokenizer_list)
    tokenizer_wrapper = GPT2TokenizerFast(
        tokenizer_object=tokenizer,
        vocab_size=args.vocab_size,
        additional_special_tokens=SPECIAL_TOKENS,
        bos_token=SPECIAL_TOKENS[0],
        eos_token=SPECIAL_TOKENS[0],
        unk_token=SPECIAL_TOKENS[0],
    )

    text = "아!@ 진짜      억울해죽것네'''아니english123배고파씌 korea으😣악😣😳😣'''"
    print(f"text: {text}")
    tokens = tokenizer_wrapper.tokenize(text)
    input_ids = tokenizer_wrapper(text)["input_ids"]
    print(f"tokens: {tokens}")
    print(f"decode: {(decoded := tokenizer_wrapper.decode(input_ids))}")
    print(f"invertible: {decoded==text}")
    tokenizer_wrapper.save_pretrained(f"{args.save_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
