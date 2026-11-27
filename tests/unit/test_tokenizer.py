import pytest
from types import SimpleNamespace

import sentencepiece as spm
from tokenizers import Tokenizer, models, pre_tokenizers

from megatron.tokenizer import build_tokenizer
from megatron.tokenizer import train_tokenizer
from megatron.tokenizer.tokenizer import HFTokenizer, SentencePieceTokenizer


@pytest.mark.cpu
def test_train_tokenizer():
    input_args = [
        "--json_input_dir",
        "./tests/data/enwik8_first100.txt",
        "--tokenizer_output_path",
        "",
    ]
    args = train_tokenizer.parse_args(input_args)
    train_tokenizer.main(args)


def _write_tokenizer_json(path, vocab):
    """Build a small WordLevel tokenizer.json with the given vocab.

    Nothing is downloaded; the file is written from the vocab dict.
    """
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(path))


def _hf_args(vocab_file, eod_token=None, pad_token=None):
    return SimpleNamespace(
        rank=0,
        tokenizer_type="HFTokenizer",
        vocab_file=str(vocab_file),
        merge_file=None,
        tokenizer_eod_token=eod_token,
        tokenizer_pad_token=pad_token,
        make_vocab_size_divisible_by=128,
        model_parallel_size=1,
    )


@pytest.mark.cpu
def test_hf_tokenizer_default_eod_and_pad(tmp_path):
    """A GPT-2-style vocab keeps the <|endoftext|> / <|padding|> defaults."""
    vocab_file = tmp_path / "tokenizer.json"
    _write_tokenizer_json(
        vocab_file,
        {"hello": 0, "world": 1, "<|endoftext|>": 2, "<|padding|>": 3, "<unk>": 4},
    )
    tokenizer = HFTokenizer(str(vocab_file))
    assert tokenizer.eod == 2
    assert tokenizer.pad_id == 3
    assert tokenizer.pad == 3


@pytest.mark.cpu
def test_hf_tokenizer_explicit_eod_and_pad_tokens(tmp_path):
    """A Gemma-style vocab (no <|endoftext|>) gets its EOD from the args."""
    vocab_file = tmp_path / "tokenizer.json"
    _write_tokenizer_json(
        vocab_file, {"hello": 0, "world": 1, "<eos>": 2, "<pad>": 3, "<unk>": 4}
    )
    # Without the explicit tokens the EOD is silently None (the #1498 bug).
    assert HFTokenizer(str(vocab_file)).eod is None
    tokenizer = HFTokenizer(str(vocab_file), eod_token="<eos>", pad_token="<pad>")
    assert tokenizer.eod == 2
    assert tokenizer.pad_id == 3
    assert tokenizer.pad == 3


@pytest.mark.cpu
def test_hf_tokenizer_explicit_token_must_be_in_vocab(tmp_path):
    vocab_file = tmp_path / "tokenizer.json"
    _write_tokenizer_json(
        vocab_file, {"hello": 0, "world": 1, "<eos>": 2, "<pad>": 3, "<unk>": 4}
    )
    with pytest.raises(ValueError, match="not in the vocab"):
        HFTokenizer(str(vocab_file), eod_token="<|endoftext|>")
    with pytest.raises(ValueError, match="not in the vocab"):
        HFTokenizer(str(vocab_file), pad_token="<|padding|>")


@pytest.mark.cpu
def test_hf_tokenizer_pad_property_without_pad_token(tmp_path):
    vocab_file = tmp_path / "tokenizer.json"
    _write_tokenizer_json(
        vocab_file, {"hello": 0, "world": 1, "<|endoftext|>": 2, "<unk>": 3}
    )
    tokenizer = HFTokenizer(str(vocab_file))
    assert tokenizer.eod == 2
    assert tokenizer.pad_id is None
    with pytest.raises(NotImplementedError):
        tokenizer.pad


@pytest.mark.cpu
def test_build_tokenizer_warns_when_default_eod_is_missing(tmp_path, capsys):
    """build_tokenizer warns (on rank 0) when <|endoftext|> is not in the vocab."""
    vocab_file = tmp_path / "tokenizer.json"
    _write_tokenizer_json(
        vocab_file, {"hello": 0, "world": 1, "<eos>": 2, "<pad>": 3, "<unk>": 4}
    )
    tokenizer = build_tokenizer(_hf_args(vocab_file))
    assert tokenizer.eod is None
    captured = capsys.readouterr()
    assert "tokenizer_eod_token" in captured.out
    assert "tokenizer_pad_token" in captured.out


@pytest.mark.cpu
def test_build_tokenizer_no_warning_when_eod_token_is_set(tmp_path, capsys):
    vocab_file = tmp_path / "tokenizer.json"
    _write_tokenizer_json(
        vocab_file, {"hello": 0, "world": 1, "<eos>": 2, "<pad>": 3, "<unk>": 4}
    )
    tokenizer = build_tokenizer(
        _hf_args(vocab_file, eod_token="<eos>", pad_token="<pad>")
    )
    assert tokenizer.eod == 2
    assert tokenizer.pad_id == 3
    captured = capsys.readouterr()
    assert "WARNING" not in captured.out


@pytest.mark.cpu
def test_build_tokenizer_rejects_eod_token_for_other_tokenizer_types():
    args = SimpleNamespace(
        rank=0,
        tokenizer_type="GPT2BPETokenizer",
        vocab_file="unused",
        merge_file="unused",
        tokenizer_eod_token="<eos>",
        tokenizer_pad_token=None,
        make_vocab_size_divisible_by=128,
        model_parallel_size=1,
    )
    with pytest.raises(ValueError, match="only supported with HFTokenizer"):
        build_tokenizer(args)


def _train_spm_model(tmp_path):
    """Train a tiny SentencePiece model whose vocab has </s> and <pad>
    but no <|endoftext|>."""
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "hello world this is a test\n" * 20 + "another line of text here\n" * 20
    )
    model_prefix = str(tmp_path / "spm")
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=model_prefix,
        vocab_size=25,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
    )
    return model_prefix + ".model"


@pytest.mark.cpu
def test_spm_tokenizer_explicit_eod_token(tmp_path):
    model_file = _train_spm_model(tmp_path)
    # Without the explicit token the EOD silently falls back to <unk> (id 0).
    tokenizer = SentencePieceTokenizer(model_file)
    assert tokenizer.eod == 0
    tokenizer = SentencePieceTokenizer(model_file, eod_token="</s>")
    assert tokenizer.eod == 2
    tokenizer = SentencePieceTokenizer(model_file, pad_token="<pad>")
    assert tokenizer.pad_id == 3
    assert tokenizer.pad == 3


@pytest.mark.cpu
def test_spm_tokenizer_explicit_token_must_be_in_vocab(tmp_path):
    model_file = _train_spm_model(tmp_path)
    with pytest.raises(ValueError, match="not in the .* vocab"):
        SentencePieceTokenizer(model_file, eod_token="<|endoftext|>")
    with pytest.raises(ValueError, match="not in the .* vocab"):
        SentencePieceTokenizer(model_file, pad_token="<|padding|>")
