from transformers import GPT2TokenizerFast, GPT2Tokenizer
import collections
from itertools import islice
import re


def consume(iterator, n):
    "Advance the iterator n-steps ahead. If n is none, consume entirely."
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_tokenizer(tokenizer_type=None, from_pretrained=True, add_padding_token=False):
    if (tokenizer_type.lower() == "hf_gpt2tokenizerfast" and from_pretrained) or tokenizer_type is None:
        tok = GPT2TokenizerFast.from_pretrained('gpt2')
        if add_padding_token:
            tok.add_special_tokens({'pad_token': '<|padding|>'})
        return tok
    elif tokenizer_type.lower() == "hf_gp2tokenizer" and from_pretrained:
        tok = GPT2Tokenizer.from_pretrained('gpt2')
        if add_padding_token:
            tok.add_special_tokens({'pad_token': '<|padding|>'})
        return tok
    else:
        raise NotImplementedError('TODO: add custom tokenizers')
