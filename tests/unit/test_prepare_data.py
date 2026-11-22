import importlib
import json
import runpy
import subprocess
import sys
from pathlib import Path
from subprocess import CalledProcessError

import pytest
import sentencepiece as spm

from megatron.data.indexed_dataset import make_dataset
from megatron.tokenizer.tokenizer import DEFAULT_EOD_TOKEN, build_tokenizer
from tools.datasets import corpora

pytestmark = pytest.mark.cpu


class DummyDownloader(corpora.DataDownloader):
    name = "dummy"
    urls = ["https://example.com/dummy.txt"]


class LocalDocuments(corpora.DataDownloader):
    name = "local-documents"
    urls = ["documents.jsonl"]
    num_workers = 1

    def download(self):
        directory = Path(self.base_dir) / self.name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / self.urls[0]).write_text('{"text": "hello"}\n{"text": "world"}\n')


def test_empty_dataset_directory_is_not_a_completed_download(tmp_path):
    downloader = DummyDownloader(
        data_dir=str(tmp_path),
        tokenizer_type="CharLevelTokenizer",
        num_workers=1,
    )
    dataset_dir = tmp_path / downloader.name
    dataset_dir.mkdir()

    assert not downloader.exists()

    (dataset_dir / "dummy.txt").write_text("data")

    assert downloader.exists()


def test_interrupted_download_can_be_retried(monkeypatch, tmp_path):
    downloader = DummyDownloader(data_dir=str(tmp_path))
    destination = tmp_path / downloader.name / "dummy.txt"

    def interrupted_download(command, *, check):
        Path(command[2]).write_text("partial data")
        raise CalledProcessError(1, command)

    monkeypatch.setattr(corpora.subprocess, "run", interrupted_download)
    with pytest.raises(RuntimeError, match="Cannot download dataset file"):
        downloader.download()

    assert not downloader.exists()
    assert list(destination.parent.iterdir()) == []

    def complete_download(command, *, check):
        Path(command[2]).write_text("complete data")

    monkeypatch.setattr(corpora.subprocess, "run", complete_download)
    monkeypatch.setattr(downloader, "tokenize", lambda: None)
    downloader.prepare()

    assert destination.read_text() == "complete data"


def test_tokenize_raises_when_preprocess_command_fails(monkeypatch, tmp_path):
    downloader = DummyDownloader(
        data_dir=str(tmp_path),
        tokenizer_type="CharLevelTokenizer",
        num_workers=1,
    )

    def failed_preprocessing(command, *, check):
        assert check
        raise CalledProcessError(1, command)

    monkeypatch.setattr(corpora.subprocess, "run", failed_preprocessing)

    with pytest.raises(RuntimeError, match="Failed to tokenize dataset dummy"):
        downloader.tokenize()


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


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize(
    ("script", "tokenizer_type", "extra_flags"),
    [
        ("preprocess_data", "HFTokenizer", []),
        (
            "preprocess_data_with_mask",
            "SPMTokenizer",
            ["--mask-before-token", "0", "--append-eod"],
        ),
    ],
)
def test_preprocessing_warns_once_for_missing_eod(
    tmp_path, gemma_like_file, spm_file, script, tokenizer_type, extra_flags, workers
):
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps({"text": "hello world"}) + "\n")
    vocab = gemma_like_file if tokenizer_type == "HFTokenizer" else spm_file
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            f"tools.datasets.{script}",
            "--input",
            str(source),
            "--output-prefix",
            str(tmp_path / "data"),
            "--tokenizer-type",
            tokenizer_type,
            "--vocab-file",
            vocab,
            "--workers",
            str(workers),
            *extra_flags,
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    warning = f"WARNING: {DEFAULT_EOD_TOKEN} is not in the {tokenizer_type} vocabulary"
    assert result.stdout.count(warning) == 1


@pytest.mark.parametrize(
    ("tokenizer_type", "eod_token", "expected_id"),
    [
        ("HFTokenizer", "<|doc end|>", 0),
        ("HFTokenizer", None, 1),
        ("SPMTokenizer", "<doc>", 3),
    ],
)
def test_prepare_data_appends_eod(
    tmp_path,
    monkeypatch,
    spm_file,
    write_hf_tokenizer,
    tokenizer_type,
    eod_token,
    expected_id,
):
    name = LocalDocuments.name
    monkeypatch.setitem(corpora.DATA_DOWNLOADERS, name, LocalDocuments)
    data_dir = tmp_path / "cached datasets"
    vocab_file = spm_file
    if tokenizer_type == "HFTokenizer":
        vocab_file = write_hf_tokenizer(
            {
                "<|doc end|>": 0,
                DEFAULT_EOD_TOKEN: 1,
                "<unk>": 2,
                "hello": 3,
                "world": 4,
            },
        )
    argv = [
        "prepare_data.py",
        name,
        "--data-dir",
        str(data_dir),
        "--tokenizer",
        tokenizer_type,
        "--vocab-file",
        vocab_file,
    ]
    if eod_token is not None:
        argv.extend(["--eod-token", eod_token])
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_module("prepare_data", run_name="__main__")

    dataset = make_dataset(
        str(data_dir / name / f"{name}_text_document"), "mmap", skip_warmup=True
    )
    assert dataset.doc_idx.tolist() == [0, 1, 2]
    assert [dataset[i][-1] for i in range(len(dataset))] == [expected_id, expected_id]


@pytest.mark.parametrize("eod_token", ["</s>", "<doc>"])
def test_masked_preprocessing_with_sentencepiece(
    spm_file, tmp_path, monkeypatch, eod_token
):
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
