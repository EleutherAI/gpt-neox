import argparse
from time import time
import logging
from datasets import load_from_disk
from transformers import GPT2TokenizerFast
from tokenizers import (
    pre_tokenizers,
    normalizers,
    decoders,
    Tokenizer,
)
from tokenizers.pre_tokenizers import (
    Punctuation,
    Digits,
    ByteLevel,
    UnicodeScripts,
)
import emoji
from tokenizers.models import BPE, Unigram
from tokenizers.trainers import BpeTrainer, UnigramTrainer
from utils import load_dataset, batch_iterator


logger = logging.getLogger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bpe", choices=["bpe", "unigram"], help="tokenizer model")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate for BPE")
    parser.add_argument("--vocab_size", type=int, default=102400, help="vocab size for tokenizer")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="tokenizer")
    args = parser.parse_args()
    return args


def main(args):
    if '.json' in args.data_path:
        dataset = load_dataset(args.data_path)
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
        "  ",
        "    ",
    ]
    with open('facial_expression.txt') as f:
        facial_expression = [line.strip() for line in f.readlines()]
    emoji_unicode_face = list(set([
        emoji.EMOJI_UNICODE['en'][i][0] for i in emoji.EMOJI_UNICODE['en'].keys() if "face" in i
    ]))
    buffer_tokens = [f"<unused{i}>" for i in range(100)]
    added_tokens = SPECIAL_TOKENS + facial_expression + emoji_unicode_face + buffer_tokens

    # tokenizer architecture
    normalizer = normalizers.NFC()
    pre_tokenizer = pre_tokenizers.Sequence([
        UnicodeScripts(), #split on different unicode range
        Punctuation(behavior = "isolated",),
        Digits(individual_digits = True),
        ByteLevel(add_prefix_space = False, use_regex = True),
    ])
    decoder = decoders.ByteLevel(add_prefix_space = False, use_regex = True)
    if args.model == "bpe":
        tokenizer = Tokenizer(BPE(dropout=args.dropout))
        trainer = BpeTrainer(
            vocab_size = args.vocab_size,
            special_tokens = added_tokens,
            initial_alphabet = pre_tokenizers.ByteLevel.alphabet(),
        )
    else:
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(
            vocab_size = args.vocab_size,
            special_tokens = added_tokens,
            initial_alphabet = pre_tokenizers.ByteLevel.alphabet(),
        )
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.decoder = decoder

    start = time()
    tokenizer.train_from_iterator(batch_iterator(dataset), trainer=trainer, length=len(dataset))
    end = time()
    print(f"time collapsed: {(end - start) / 60:.2f}m")
    tokenizer_wrapper = GPT2TokenizerFast(
        tokenizer_object = tokenizer,
        vocab_size = args.vocab_size,
        additional_special_tokens = SPECIAL_TOKENS,
        bos_token = SPECIAL_TOKENS[0],
        eos_token = SPECIAL_TOKENS[0],
        unk_token = SPECIAL_TOKENS[0],
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