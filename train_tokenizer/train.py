import argparse
from time import time
import logging
from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast
from tokenizers import (
    normalizers,
    decoders,
    Tokenizer,
    processors,
    Regex,
)
from tokenizers.pre_tokenizers import (
    Punctuation,
    Digits,
    ByteLevel,
    UnicodeScripts,
    Sequence,
    Split,
    WhitespaceSplit,
    Whitespace,
)
import emoji
from tokenizers.models import BPE, Unigram
from tokenizers.trainers import BpeTrainer, UnigramTrainer
from utils import load_dataset, batch_iterator, load_from_path
import os
import regex as re
import datasets

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
    parser.add_argument("--dropout", default=None, help="dropout rate for BPE")
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
        default="NFC",
        choices=["NFKC", "NFC"],
        help="unicode normalizer",
    )
    parser.add_argument(
        "--byte_fallback",
        type=bool,
        default=False,
        choices=[True, False],
        help="Bytelevel() preprocessor vs BPE(byte_fallback)",
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
        "--whitespace_reservation",
        type=int,
        default=24,
        help="number of whitespaces to add as special tokens. \n \
            default length linear. \n \
            sorted down from len = (whitespace_reservation)",
        # consider no repeat ngrams during generation. (3 indentations--> bad)
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
    parser.add_argument(
        "--remove_longspace",
        type=bool,
        default=False,
        choices=[True, False],
        help="during tokenizer training preprocessing, remove long whitespaces(longer than 16)",
    )
    parser.add_argument(
        "--single_whitespace",
        type=bool,
        default=False,
        choices=[True, False],
        help="Whether to include single whitespace in vocab",
    )
    parser.add_argument(
        "--add_prefix_space",
        type=bool,
        default=True,
        choices=[True, False],
        help="add prefix space. True : 'Gword','word' ",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=0,
        help="corpus shuffle seed. 0(default) for no shuffling",
    )
    args, _ = parser.parse_known_args()
    return args


def main(args):
    data_path = args.data_path

    dataset = load_from_path(data_path)
    if dataset == None:
        raise ValueError(f"Check --data_path : {data_path}")
    if seed := args.shuffle_seed != 0:
        dataset.shuffle(seed)

    # tokenizer arguments
    SPECIAL_TOKENS = [
        "<s>",
        "</s>",
        "<|usr|>",
        "<|pad|>",
        "<|sys|>",
        "<|unk|>",
        "<|sep|>",
        "<|mask|>",
    ]  # TODO : add specific tokens. add || https://github.com/EleutherAI/dps/blob/master/dps/spark/utils/token_utils.py
    # calculate whitespace tokens
    whitespace = " "
    whitespace_count = args.whitespace_reservation  # 4,2 whitespaces

    # construct whitespaces
    whitespace_list = [whitespace * count for count in range(whitespace_count, 1, -1)]
    if args.single_whitespace:
        whitespace_list.append(" ")  # add single_whitespace
    # necessary for invertibility?
    vocab_size = args.vocab_size - len(whitespace_list)  # we will add whitespace later
    # construct buffer_tokens
    buffer_token_count = args.buffer_tokens
    buffer_tokens = [f"<|unused{i}|>" for i in range(buffer_token_count)]
    # construct added_token
    added_tokens = (
        SPECIAL_TOKENS
        + buffer_tokens
        # + whitespace_list #not a special token
    )
    add_prefix_space = args.add_prefix_space

    # tokenizer normalizer
    if args.normalizer.lower() == "nfc":
        normalizer = normalizers.NFC()
    elif args.normalizer.lower() == "nfkc":
        normalizer = normalizers.NFKC()

    # use Split() to prevent long spaces. allow up to (17 - 1) whitespace tokens
    split_regex = re.compile(r"\s{17,}", cache_pattern=True)
    split_pattern = Regex(split_regex.pattern)

    # common pretokenizer
    pre_tokenizer_list = [
        UnicodeScripts(),  # split on different unicode range
        Punctuation(
            behavior="isolated",  # not contiguous /* */  /*******/
        ),
        Digits(individual_digits=True),
    ]
    if args.remove_longspace == True:
        pre_tokenizer_list.append(
            Split(pattern=split_pattern, behavior="isolated", invert=False)
        )
    pre_tokenizer_list.append(ByteLevel(add_prefix_space=False, use_regex=True))

    # set byte_fallback
    byte_fallback = args.byte_fallback

    # if yes, default to no whitespace handling
    if not args.preserve_whitespace == "yes":
        pre_tokenizer_list.insert(
            0,
            Whitespace(),  # WhitespaceSplit()
        )  # whitespace split should be in front
    # common decoder
    decoder = decoders.ByteLevel(add_prefix_space=False, use_regex=True)
    # construct_pretokenizer
    pre_tokenizer = Sequence(pre_tokenizer_list)

    if args.model.lower() == "bpe":
        tokenizer = Tokenizer(
            BPE(
                cache_capacity=args.cache_capacity,
                dropout=args.dropout,
                byte_fallback=byte_fallback,
            )
        )

        # change behavior for byte_fallback
        if byte_fallback:
            initial_alphabet = []
            decoder = decoders.ByteFallback()
            for idx in range(len(pre_tokenizer_list)):
                if isinstance(pre_tokenizer_list[idx], ByteLevel):
                    pre_tokenizer_list.pop(idx)
                    break
            pre_tokenizer = Sequence(pre_tokenizer_list)
        else:
            initial_alphabet = ByteLevel.alphabet()
            initial_alphabet.sort()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=added_tokens,
            initial_alphabet=initial_alphabet,
            continuing_subword_prefix=args.continuing_subword_prefix,  # ##word
        )
    else:
        initial_alphabet = ByteLevel.alphabet()
        initial_alphabet.sort()
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=added_tokens,
            initial_alphabet=ByteLevel.alphabet(),
        )
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.decoder = decoder

    start = time()
    if isinstance(dataset, datasets.arrow_dataset.Dataset):
        tokenizer.train_from_iterator(
            batch_iterator(dataset), trainer=trainer, length=len(dataset)
        )

    end = time()
    print(f"time elapsed: {(end - start) / 60:.2f}m")
    # if preserve whitespace is set to "inference", remove Whitespace splitter
    if args.preserve_whitespace == "inference":
        for item in pre_tokenizer_list:
            if isinstance(item, Whitespace):
                pre_tokenizer_list.remove(item)
                break

    if args.remove_longspace == True:
        for item in pre_tokenizer_list:
            if isinstance(item, Split):
                pre_tokenizer_list.remove(item)
                break
    tokenizer.pre_tokenizer = Sequence(pre_tokenizer_list)
    tokenizer.add_tokens(whitespace_list)

    # wrap tokenizer

    tokenizer_wrapper = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        vocab_size=args.vocab_size,
        additional_special_tokens=added_tokens,
        bos_token=SPECIAL_TOKENS[0],  # GPT style all [0]
        eos_token=SPECIAL_TOKENS[0],  #
        unk_token=SPECIAL_TOKENS[0],  #
    )

    text = "아!@ ㈝12시 진짜홀수ws     tOkeN  짝수ws    억울해죽것네'''newLine\nNewline taB\tTab 아니enGlish123배고파씌 Korea으😣악😣😳😣'''"

    tokens = tokenizer_wrapper.tokenize(text)
    input_ids = tokenizer_wrapper(text)["input_ids"]
    vocab_dict_vk = {v: k for k, v in tokenizer_wrapper.vocab.items()}
    decoded_dict = {}
    for idx, token in vocab_dict_vk.items():
        decoded_dict[idx] = tokenizer.decoder.decode([token])
    each_decode = []
    ufb_count = 0
    for id in input_ids:
        decoded = decoded_dict[id]
        if decoded == "�":
            each_decode.append("|ufb|")
            ufb_count += 1
        else:
            each_decode.append(decoded)

    print(f"text:{text}")
    print(f"decode:{(decoded := tokenizer_wrapper.decode(input_ids))}")
    print(f"invertible: {decoded==text}")
    print(f"tokens: {tokens}")
    print(f"tokens recon: {each_decode}")
    print(
        f"original input length: char= {len(text)}, bytes= {len(text.encode('utf-8'))}"
    )
    print(f"token count: {len(tokens)}")
    print(
        f"unicode fallback portion: {ufb_count} /{len(tokens)} = {ufb_count / len(tokens):.3f}"
    )

    tokenizer_wrapper.save_pretrained(f"{args.save_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
