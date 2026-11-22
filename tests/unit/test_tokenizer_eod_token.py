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
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import sentencepiece as spm
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers, processors

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
        user_defined_symbols=["<doc>"],
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


@pytest.mark.parametrize(
    ("tokenizer_type", "eod_token", "expected_id"),
    [
        ("HFTokenizer", "<|doc end|>", 0),
        ("HFTokenizer", None, 1),
        ("SPMTokenizer", "<doc>", 3),
    ],
)
def test_prepare_data_appends_eod(
    tmp_path, spm_file, tokenizer_type, eod_token, expected_id
):
    import subprocess
    import sys
    from pathlib import Path
    from zipfile import ZipFile

    from megatron.data.indexed_dataset import make_dataset

    data_dir = tmp_path / "cached datasets"
    corpus = data_dir / "enwik8"
    corpus.mkdir(parents=True)
    with ZipFile(corpus / "enwik8.zip", "w") as archive:
        archive.writestr("first.txt", "hello")
        archive.writestr("second.txt", "world")

    vocab_file = spm_file
    if tokenizer_type == "HFTokenizer":
        vocab_file = _hf_tokenizer_file(
            tmp_path,
            {
                "<|doc end|>": 0,
                DEFAULT_EOD_TOKEN: 1,
                "<unk>": 2,
                "hello": 3,
                "world": 4,
            },
        )
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[2] / "prepare_data.py"),
        "enwik8",
        "--data-dir",
        str(data_dir),
        "--tokenizer",
        tokenizer_type,
        "--vocab-file",
        vocab_file,
    ]
    if eod_token is not None:
        cmd.extend(["--eod-token", eod_token])
    subprocess.run(cmd, check=True)

    dataset = make_dataset(
        str(corpus / "enwik8_text_document"), "mmap", skip_warmup=True
    )
    assert dataset.doc_idx.tolist() == [0, 1, 2]
    assert [dataset[i][-1] for i in range(len(dataset))] == [expected_id, expected_id]


@pytest.mark.parametrize("eod_token", ["</s>", "<doc>"])
def test_masked_preprocessing_with_sentencepiece(
    spm_file, tmp_path, monkeypatch, eod_token
):
    import sys

    from megatron.data.indexed_dataset import make_dataset
    from tools.datasets import preprocess_data_with_mask

    text = "hello world"
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps({"text": text}) + "\n")
    sp = spm.SentencePieceProcessor(model_file=spm_file)
    delimiter = sp.encode("hello")
    prefix = tmp_path / "data"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preprocess_data_with_mask",
            "--input",
            str(source),
            "--output-prefix",
            str(prefix),
            "--tokenizer-type",
            "SPMTokenizer",
            "--vocab-file",
            spm_file,
            "--mask-before-token",
            ",".join(map(str, delimiter)),
            "--append-eod",
            "--eod-token",
            eod_token,
        ],
    )
    preprocess_data_with_mask.main()

    tokens = sp.encode(text) + [sp.piece_to_id(eod_token)]
    text_rows = make_dataset(f"{prefix}_text_document", "mmap", skip_warmup=True)
    label_rows = make_dataset(f"{prefix}_label_document", "mmap", skip_warmup=True)
    assert text_rows[0].tolist() == tokens
    assert label_rows[0].tolist() == [-100] * len(delimiter) + tokens[len(delimiter) :]


@pytest.fixture
def hf_export_config(gemma_like_file):
    return {
        "tokenizer-type": "HFTokenizer",
        "vocab-file": gemma_like_file,
        "make-vocab-size-divisible-by": 1,
        "model-parallel-size": 1,
        "hidden-size": 8,
        "num-layers": 1,
        "num-attention-heads": 2,
        "max-position-embeddings": 16,
    }


@pytest.mark.parametrize(
    ("architecture", "eod_config", "expected_eos_id"),
    [
        ("neox", {"tokenizer_eod_token": "<eos>"}, 1),
        ("neox", {"tokenizer-eod-token": "<pad>"}, 0),
        ("llama", {"tokenizer_eod_token": "<eos>"}, 1),
    ],
)
def test_hf_export_eos_token_id(
    hf_export_config, tmp_path, architecture, eod_config, expected_eos_id
):
    from transformers import AutoConfig

    from tools.ckpts import convert_neox_to_hf

    hf_export_config.update(eod_config)
    hf_config = convert_neox_to_hf.create_config(
        hf_export_config, architecture=architecture
    )

    hf_config.save_pretrained(tmp_path)
    assert AutoConfig.from_pretrained(tmp_path).eos_token_id == expected_eos_id


def test_neox_export_uses_architecture_default_when_training_eod_is_unset(
    hf_export_config,
):
    from transformers import GPTNeoXConfig
    from tools.ckpts.convert_neox_to_hf import create_config

    assert HFTokenizer(hf_export_config["vocab-file"]).eod is None
    assert create_config(hf_export_config).eos_token_id == GPTNeoXConfig().eos_token_id


@pytest.fixture
def run_export(tmp_path):
    from tools.ckpts import convert_neox_to_hf

    def export(config, *flags):
        config_file = tmp_path / "neox.json"
        config_file.write_text(json.dumps(config))
        output = tmp_path / "export"
        convert_neox_to_hf.main(
            ["--config_file", str(config_file), "--output_dir", str(output), *flags]
        )
        return output

    return export


@pytest.fixture
def export_with_tokenizer(run_export, monkeypatch):
    from transformers import AutoModelForCausalLM

    def convert_without_checkpoint(input_dir, config, output_dir, **kwargs):
        return AutoModelForCausalLM.from_config(kwargs["hf_config"])

    monkeypatch.setattr(
        "tools.ckpts.convert_neox_to_hf.convert", convert_without_checkpoint
    )
    return run_export


@pytest.fixture
def export_with_invalid_tokenizer(run_export, monkeypatch):
    def unexpected_conversion(*args, **kwargs):
        pytest.fail(
            "Checkpoint conversion started before the export tokenizer was validated"
        )

    monkeypatch.setattr("tools.ckpts.convert_neox_to_hf.convert", unexpected_conversion)
    return run_export


@pytest.mark.parametrize("pretrained", [False, True], ids=["raw-json", "pretrained"])
@pytest.mark.parametrize("eod_token", ["<eos>", "<pad>", None])
def test_saved_tokenizer_eos(
    hf_export_config, tmp_path, export_with_tokenizer, pretrained, eod_token
):
    from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerFast

    vocab_file = hf_export_config["vocab-file"]
    backend = Tokenizer.from_file(vocab_file)
    backend.add_tokens([AddedToken("<eos>", lstrip=True, normalized=False)])
    backend.save(vocab_file)
    flags = ["--pad-token-id", "0"]
    if pretrained:
        PreTrainedTokenizerFast(
            tokenizer_file=vocab_file,
            eos_token="world",
            bos_token="<bos>",
        ).save_pretrained(tmp_path)
        flags.append("--vocab-is-hf-tokenizer")
    if eod_token is not None:
        hf_export_config["tokenizer-eod-token"] = eod_token

    output = export_with_tokenizer(hf_export_config, *flags)
    tokenizer = AutoTokenizer.from_pretrained(output)
    assert tokenizer.get_vocab() == GEMMA_LIKE_VOCAB
    assert tokenizer.pad_token_id == 0
    if eod_token is None:
        assert tokenizer.eos_token == ("world" if pretrained else None)
    else:
        assert tokenizer.eos_token == eod_token
        eos_id = tokenizer.eos_token_id
        assert eos_id == AutoConfig.from_pretrained(output).eos_token_id
        assert eos_id == GEMMA_LIKE_VOCAB[eod_token]
        assert tokenizer.decode([eos_id], skip_special_tokens=True) == ""


@pytest.mark.parametrize(
    ("architecture", "source"),
    [
        ("neox", "raw-hf"),
        ("neox", "pretrained-hf"),
        ("llama", "sentencepiece"),
        ("mistral", "sentencepiece"),
    ],
)
def test_export_of_explicit_default_matches_implicit_eod(
    hf_export_config, spm_file, tmp_path, export_with_tokenizer, architecture, source
):
    from transformers import (
        AutoConfig,
        AutoTokenizer,
        GenerationConfig,
        LlamaTokenizerFast,
        PreTrainedTokenizerFast,
    )

    flags = ["--architecture", architecture]
    if source == "sentencepiece":
        sp = spm.SentencePieceProcessor(model_file=spm_file)
        vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
        equivalent_eod = "<unk>"
        hf_export_config.update(
            {"tokenizer-type": "SPMTokenizer", "vocab-file": spm_file}
        )
        vocab_file = _hf_tokenizer_file(tmp_path, vocab)
        LlamaTokenizerFast(
            tokenizer_file=vocab_file,
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            add_eos_token=True,
        ).save_pretrained(tmp_path)
    else:
        vocab = {DEFAULT_EOD_TOKEN: 0, "<unk>": 1, "hello": 2, "world": 3}
        equivalent_eod = DEFAULT_EOD_TOKEN
        vocab_file = _hf_tokenizer_file(tmp_path, vocab)
        if source == "pretrained-hf":
            PreTrainedTokenizerFast(
                tokenizer_file=vocab_file, eos_token="world"
            ).save_pretrained(tmp_path)

    if source != "raw-hf":
        flags.append("--vocab-is-hf-tokenizer")

    for eod_config in ({}, {"tokenizer_eod_token": equivalent_eod}):
        output = export_with_tokenizer({**hf_export_config, **eod_config}, *flags)
        for config in (
            AutoConfig.from_pretrained(output),
            GenerationConfig.from_pretrained(output),
        ):
            assert (config.bos_token_id, config.eos_token_id) == (0, 0)
        saved = AutoTokenizer.from_pretrained(output)
        assert (saved.eos_token, saved.eos_token_id) == (equivalent_eod, 0)
        assert saved.decode([0], skip_special_tokens=True) == ""
        if source == "sentencepiece":
            assert saved.bos_token_id == sp.bos_id()
            backend = Tokenizer.from_file(str(output / "tokenizer.json"))
            assert backend.encode("h").ids == [sp.bos_id(), sp.piece_to_id("h"), 0]


@pytest.mark.parametrize("explicit", [False, True])
def test_export_rejects_different_eos_ids(
    hf_export_config, tmp_path, export_with_invalid_tokenizer, explicit
):
    from transformers import PreTrainedTokenizerFast

    # The NeoX vocab file and the HF directory can contain different tokenizers.
    training_vocab = {DEFAULT_EOD_TOKEN: 0, "<unk>": 1, "hello": 2}
    _hf_tokenizer_file(tmp_path, training_vocab)
    training_file = (tmp_path / "tokenizer.json").rename(
        tmp_path / "training-vocab.json"
    )
    hf_vocab = {**training_vocab, DEFAULT_EOD_TOKEN: 2, "hello": 0}
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel(vocab=hf_vocab, unk_token="<unk>"))
    ).save_pretrained(tmp_path)
    hf_export_config["vocab-file"] = str(training_file)
    if explicit:
        hf_export_config["tokenizer_eod_token"] = DEFAULT_EOD_TOKEN

    with pytest.raises(ValueError, match="has ID 2 .*model expects 0"):
        export_with_invalid_tokenizer(hf_export_config, "--vocab-is-hf-tokenizer")
    assert not (tmp_path / "export").exists()


def test_unreadable_pretrained_tokenizer_fails_before_conversion(
    hf_export_config, tmp_path, export_with_invalid_tokenizer
):
    (tmp_path / "tokenizer_config.json").write_text("{")
    with pytest.raises(json.JSONDecodeError):
        export_with_invalid_tokenizer(hf_export_config, "--vocab-is-hf-tokenizer")
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize(
    ("architecture", "default_bos", "default_eos"),
    [("llama", None, None), ("mistral", None, None), ("neox", 0, 2)],
)
@pytest.mark.parametrize("pretrained", [False, True], ids=["raw-json", "pretrained"])
@pytest.mark.parametrize("bos_id", [None, 2], ids=["default-bos", "explicit-bos"])
def test_export_bos_is_independent_of_eod(
    hf_export_config,
    tmp_path,
    export_with_tokenizer,
    architecture,
    default_bos,
    default_eos,
    pretrained,
    bos_id,
):
    from transformers import (
        AutoConfig,
        AutoTokenizer,
        GenerationConfig,
        PreTrainedTokenizerFast,
    )

    # NeoX encodes a Gemma-style prefix even if the HF metadata declares a different BOS.
    vocab_file = hf_export_config["vocab-file"]
    backend = Tokenizer.from_file(vocab_file)
    backend.post_processor = processors.TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", 2)]
    )
    PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<pad>", eos_token="world"
    ).save_pretrained(tmp_path)
    assert HFTokenizer(vocab_file).tokenize("hello world") == [2, 4, 5]

    flags = ["--architecture", architecture]
    if pretrained:
        flags.append("--vocab-is-hf-tokenizer")
    expected_tokenizer_bos = 0 if pretrained else None
    if bos_id is not None:
        flags.extend(["--bos-token-id", str(bos_id)])
        expected_tokenizer_bos = bos_id

    expected_bos = default_bos if bos_id is None else bos_id
    for eod_config, expected_eos in [
        ({}, default_eos),
        ({"tokenizer_eod_token": "<eos>"}, 1),
    ]:
        output = export_with_tokenizer({**hf_export_config, **eod_config}, *flags)
        for config in (
            AutoConfig.from_pretrained(output),
            GenerationConfig.from_pretrained(output),
        ):
            assert (config.bos_token_id, config.eos_token_id) == (
                expected_bos,
                expected_eos,
            )
        tokenizer = AutoTokenizer.from_pretrained(output)
        assert tokenizer.bos_token_id == expected_tokenizer_bos
        assert tokenizer.encode("hello world") == [2, 4, 5]
        if eod_config:
            assert tokenizer.eos_token_id == expected_eos


@pytest.mark.parametrize("architecture", ["llama", "mistral"])
def test_spm_export_preserves_legacy_bos(hf_export_config, spm_file, architecture):
    from tools.ckpts.convert_neox_to_hf import create_config

    hf_export_config.update(
        {
            "tokenizer-type": "SPMTokenizer",
            "vocab-file": spm_file,
        }
    )
    legacy = create_config(hf_export_config, architecture=architecture)
    assert (legacy.bos_token_id, legacy.eos_token_id) == (0, 0)
    hf_export_config["tokenizer_eod_token"] = "</s>"
    config = create_config(hf_export_config, architecture=architecture)
    assert (config.bos_token_id, config.eos_token_id) == (0, 2)
    explicit = create_config(
        hf_export_config, architecture=architecture, bos_token_id=1
    )
    assert (explicit.bos_token_id, explicit.eos_token_id) == (1, 2)


def test_raw_export_preserves_legacy_bos(hf_export_config, neox_like_file):
    from tools.ckpts.convert_neox_to_hf import create_config

    hf_export_config["vocab-file"] = neox_like_file
    legacy = create_config(hf_export_config, architecture="llama")
    assert (legacy.bos_token_id, legacy.eos_token_id) == (0, 0)
    hf_export_config["tokenizer_eod_token"] = "world"
    custom = create_config(hf_export_config, architecture="llama")
    assert (custom.bos_token_id, custom.eos_token_id) == (0, 3)


@pytest.mark.parametrize("bos_id", [None, 0])
def test_model_only_bos_ignores_custom_tokenizer_class(
    hf_export_config, tmp_path, export_with_tokenizer, bos_id
):
    from transformers import AutoConfig, GenerationConfig

    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "CustomTokenizer",
                "auto_map": {
                    "AutoTokenizer": ["tokenization_custom.CustomTokenizer", None]
                },
            }
        )
    )
    hf_export_config["tokenizer_eod_token"] = "<eos>"
    flags = ["--bos-token-id", str(bos_id)] if bos_id is not None else []
    output = export_with_tokenizer(
        hf_export_config,
        "--architecture",
        "llama",
        "--vocab-is-hf-tokenizer",
        "--no_save_tokenizer",
        *flags,
    )
    assert AutoConfig.from_pretrained(output).bos_token_id == bos_id
    assert GenerationConfig.from_pretrained(output).bos_token_id == bos_id
    assert not (output / "tokenizer.json").exists()


def test_explicit_bos_updates_pretrained_tokenizer_template(
    hf_export_config, tmp_path, export_with_tokenizer
):
    from transformers import AutoConfig, AutoTokenizer, LlamaTokenizerFast

    vocab_file = _hf_tokenizer_file(
        tmp_path, {"<start>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "hello": 4}
    )
    LlamaTokenizerFast(tokenizer_file=vocab_file).save_pretrained(tmp_path)
    output = export_with_tokenizer(
        hf_export_config,
        "--architecture",
        "mistral",
        "--bos-token-id",
        "0",
        "--vocab-is-hf-tokenizer",
    )
    assert AutoConfig.from_pretrained(output).bos_token_id == 0
    tokenizer = AutoTokenizer.from_pretrained(output)
    assert tokenizer.bos_token == "<start>"
    assert tokenizer.encode("hello") == [0, 4]
    assert Tokenizer.from_file(str(output / "tokenizer.json")).encode("hello").ids == [
        0,
        4,
    ]


@pytest.mark.parametrize("bos_id", [-1, len(GEMMA_LIKE_VOCAB)])
def test_invalid_explicit_bos(hf_export_config, export_with_tokenizer, bos_id):
    hf_export_config["make-vocab-size-divisible-by"] = 128
    with pytest.raises(ValueError, match="outside the tokenizer vocabulary"):
        export_with_tokenizer(
            hf_export_config, "--bos-token-id", str(bos_id), "--no_save_tokenizer"
        )


def test_explicit_bos_must_exist_in_saved_tokenizer(
    hf_export_config, tmp_path, export_with_invalid_tokenizer
):
    from transformers import PreTrainedTokenizerFast

    source = (tmp_path / "tokenizer.json").rename(tmp_path / "training-vocab.json")
    hf_export_config["vocab-file"] = str(source)
    small_vocab = _hf_tokenizer_file(tmp_path, {"<unk>": 0, "hello": 1})
    PreTrainedTokenizerFast(tokenizer_file=small_vocab).save_pretrained(tmp_path)
    with pytest.raises(
        ValueError, match="--bos-token-id 5 is missing from the export tokenizer"
    ):
        export_with_invalid_tokenizer(
            hf_export_config, "--bos-token-id", "5", "--vocab-is-hf-tokenizer"
        )
    assert not (tmp_path / "export").exists()
