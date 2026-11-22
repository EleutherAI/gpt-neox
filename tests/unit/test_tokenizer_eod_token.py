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

from copy import deepcopy
from types import SimpleNamespace

import pytest
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


@pytest.mark.parametrize("warn_missing_eod", [True, False])
@pytest.mark.parametrize(
    ("flag", "name"),
    [("eod_mask_loss", "eod_mask_loss"), ("append_eod", "--append-eod")],
)
def test_missing_eod_raises_when_it_is_needed(
    gemma_like_file, flag, name, warn_missing_eod
):
    with pytest.raises(
        ValueError, match=f"{name} needs an end-of-document token"
    ) as exc:
        build_tokenizer(
            _args("HFTokenizer", gemma_like_file, **{flag: True}),
            warn_missing_eod=warn_missing_eod,
        )
    assert "tokenizer_eod_token" in str(exc.value)
    assert "--eod-token" in str(exc.value)


@pytest.mark.parametrize(
    ("flag", "name"),
    [("eod_mask_loss", "eod_mask_loss"), ("append_eod", "--append-eod")],
)
def test_hfgpt2_missing_eos_uses_pretrained_tokenizer_settings(tmp_path, flag, name):
    from transformers import GPT2TokenizerFast

    backend = Tokenizer(models.BPE(vocab={DEFAULT_EOD_TOKEN: 0, "a": 1}, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel()
    source = GPT2TokenizerFast(tokenizer_object=backend, eos_token=None)
    source.save_pretrained(tmp_path)

    args = _args("HFGPT2Tokenizer", str(tmp_path))
    assert build_tokenizer(args).eod is None
    setattr(args, flag, True)
    with pytest.raises(
        ValueError, match=f"{name} needs an end-of-document token"
    ) as exc:
        build_tokenizer(args)
    message = str(exc.value)
    assert "HFGPT2Tokenizer" in message
    assert "eos_token" in message
    assert DEFAULT_EOD_TOKEN not in message
    assert "tokenizer_eod_token" not in message
    assert "--eod-token" not in message

    source.eos_token = DEFAULT_EOD_TOKEN
    source.save_pretrained(tmp_path)
    assert build_tokenizer(args).eod == 0


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


@pytest.mark.parametrize("tokenizer_type", ["CharLevelTokenizer", "HFGPT2Tokenizer"])
def test_eod_token_is_rejected_for_other_tokenizers(tokenizer_type):
    with pytest.raises(
        ValueError, match="only supported with HFTokenizer and SPMTokenizer"
    ):
        build_tokenizer(_args(tokenizer_type, None, tokenizer_eod_token="<eos>"))


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
