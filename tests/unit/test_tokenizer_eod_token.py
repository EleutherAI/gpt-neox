# Copyright (c) 2026, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
from copy import deepcopy
from types import SimpleNamespace

import pytest
import sentencepiece as spm
from tokenizers import Tokenizer, models, pre_tokenizers

from megatron.neox_arguments import NeoXArgs
from megatron.tokenizer.tokenizer import (
    DEFAULT_EOD_TOKEN,
    HFTokenizer,
    SentencePieceTokenizer,
    build_tokenizer,
)
from tests.common import BASE_CONFIG

pytestmark = pytest.mark.cpu

# Laid out like Gemma's special tokens: no <|endoftext|>, EOS is <eos>
GEMMA_LIKE_VOCAB = {
    "<pad>": 0,
    "<eos>": 1,
    "<bos>": 2,
    "<unk>": 3,
    "hello": 4,
    "world": 5,
}


def _hf_tokenizer_file(tmp_path, vocab):
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    return str(path)


@pytest.fixture
def gemma_like_file(tmp_path):
    return _hf_tokenizer_file(tmp_path, GEMMA_LIKE_VOCAB)


@pytest.fixture
def neox_like_file(tmp_path):
    vocab = {"<|endoftext|>": 0, "<|padding|>": 1, "hello": 2, "world": 3, "<unk>": 4}
    return _hf_tokenizer_file(tmp_path, vocab)


@pytest.fixture
def spm_file(tmp_path):
    # SentencePiece's default special pieces: <unk> = 0, <s> = 1, </s> = 2
    model = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(["hello world", "hello there world"] * 50),
        model_writer=model,
        vocab_size=20,
        model_type="char",
    )
    path = tmp_path / "tokenizer.model"
    path.write_bytes(model.getvalue())
    return str(path)


def _args(tokenizer_type, vocab_file, **kwargs):
    return SimpleNamespace(
        rank=0,
        tokenizer_type=tokenizer_type,
        vocab_file=vocab_file,
        merge_file=None,
        make_vocab_size_divisible_by=1,
        model_parallel_size=1,
        **kwargs,
    )


def test_hf_default_eod_is_unchanged(neox_like_file, capsys):
    tokenizer = build_tokenizer(_args("HFTokenizer", neox_like_file))

    assert tokenizer.eod == 0
    assert tokenizer.pad_id == 1
    assert "WARNING" not in capsys.readouterr().out


def test_hf_explicit_eod_token(gemma_like_file, capsys):
    tokenizer = build_tokenizer(
        _args("HFTokenizer", gemma_like_file, tokenizer_eod_token="<eos>")
    )

    assert tokenizer.eod == 1
    assert "WARNING" not in capsys.readouterr().out


def test_hf_explicit_eod_token_overrides_default(neox_like_file):
    tokenizer = HFTokenizer(neox_like_file, eod_token="world")

    assert tokenizer.eod == 3


def test_hf_unknown_eod_token_raises(gemma_like_file):
    with pytest.raises(ValueError, match="'<eot>' is not in the vocabulary"):
        HFTokenizer(gemma_like_file, eod_token="<eot>")


def test_hf_missing_default_eod_warns(gemma_like_file, capsys):
    tokenizer = build_tokenizer(_args("HFTokenizer", gemma_like_file))

    assert tokenizer.eod is None
    out = capsys.readouterr().out
    assert f"WARNING: {DEFAULT_EOD_TOKEN} is not in the HFTokenizer vocabulary" in out
    assert "the end-of-document token is unset" in out
    assert "tokenizer_eod_token" in out


def test_warning_is_only_printed_on_rank_0(gemma_like_file, capsys):
    args = _args("HFTokenizer", gemma_like_file)
    args.rank = 1

    build_tokenizer(args)

    assert "WARNING" not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("flag", "name"),
    [("eod_mask_loss", "eod_mask_loss"), ("append_eod", "--append-eod")],
)
def test_missing_eod_raises_when_it_is_needed(gemma_like_file, flag, name):
    with pytest.raises(ValueError, match=f"{name} needs an end-of-document token"):
        build_tokenizer(_args("HFTokenizer", gemma_like_file, **{flag: True}))


@pytest.mark.parametrize("flag", ["eod_mask_loss", "append_eod"])
def test_configured_eod_satisfies_flags_that_need_it(gemma_like_file, flag):
    tokenizer = build_tokenizer(
        _args(
            "HFTokenizer", gemma_like_file, tokenizer_eod_token="<eos>", **{flag: True}
        )
    )

    assert tokenizer.eod == 1


def test_spm_explicit_eod_token(spm_file, capsys):
    tokenizer = build_tokenizer(
        _args("SPMTokenizer", spm_file, tokenizer_eod_token="</s>")
    )

    assert tokenizer.eod == 2
    assert "WARNING" not in capsys.readouterr().out


def test_spm_missing_default_eod_keeps_unknown_token_and_warns(spm_file, capsys):
    tokenizer = build_tokenizer(_args("SPMTokenizer", spm_file))

    # piece_to_id maps a missing piece to <unk>; that stays the EOD token
    assert tokenizer.eod == tokenizer.tokenizer.unk_id() == 0
    out = capsys.readouterr().out
    assert f"WARNING: {DEFAULT_EOD_TOKEN} is not in the SPMTokenizer vocabulary" in out
    assert "the end-of-document token is the unknown token" in out


def test_spm_unknown_eod_token_raises(spm_file):
    with pytest.raises(ValueError, match="'<eos>' is not in the vocabulary"):
        SentencePieceTokenizer(spm_file, eod_token="<eos>")


def test_eod_token_is_rejected_for_other_tokenizers():
    with pytest.raises(
        ValueError, match="only supported with HFTokenizer and SPMTokenizer"
    ):
        build_tokenizer(_args("CharLevelTokenizer", None, tokenizer_eod_token="<eos>"))


def test_neox_args_pass_the_eod_token_to_the_tokenizer(gemma_like_file):
    config = deepcopy(BASE_CONFIG)
    config.update(
        global_num_gpus=1,
        tokenizer_type="HFTokenizer",
        vocab_file=gemma_like_file,
        tokenizer_eod_token="<eos>",
        eod_mask_loss=True,
    )
    neox_args = NeoXArgs.from_dict(config)

    neox_args.build_tokenizer()

    assert neox_args.tokenizer.eod == 1


@pytest.mark.parametrize(
    ("script", "reads_sys_argv"),
    [
        ("tools.datasets.preprocess_data", False),
        # preprocess_data_with_mask.get_args takes no arguments
        ("tools.datasets.preprocess_data_with_mask", True),
    ],
)
def test_preprocessing_eod_token_flag(
    gemma_like_file, script, reads_sys_argv, monkeypatch
):
    import importlib
    import sys

    get_args = importlib.import_module(script).get_args
    argv = [
        "--input",
        "unused.jsonl",
        "--output-prefix",
        "unused",
        "--tokenizer-type",
        "HFTokenizer",
        "--vocab-file",
        gemma_like_file,
        "--append-eod",
        "--eod-token",
        "<eos>",
    ]
    if reads_sys_argv:
        monkeypatch.setattr(sys, "argv", ["preprocess"] + argv)
        args = get_args()
    else:
        args = get_args(argv)

    assert args.tokenizer_eod_token == "<eos>"
    assert build_tokenizer(args).eod == 1


def test_hf_export_uses_the_eod_token(gemma_like_file):
    from tools.ckpts import convert_neox_to_hf

    hf_config = convert_neox_to_hf.create_config(
        {
            "tokenizer-type": "HFTokenizer",
            "vocab-file": gemma_like_file,
            "tokenizer_eod_token": "<eos>",
            "make-vocab-size-divisible-by": 1,
            "model-parallel-size": 1,
            "hidden-size": 8,
            "num-layers": 1,
            "num-attention-heads": 2,
            "max-position-embeddings": 16,
            "norm": "rmsnorm",
        },
        architecture="llama",
    )

    assert hf_config.eos_token_id == 1
