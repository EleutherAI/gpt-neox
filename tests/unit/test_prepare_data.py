from pathlib import Path
from subprocess import CalledProcessError

import pytest

from tools.datasets import corpora


class DummyDownloader(corpora.DataDownloader):
    name = "dummy"
    urls = ["https://example.com/dummy.txt"]


@pytest.mark.cpu
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


@pytest.mark.cpu
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


@pytest.mark.cpu
def test_tokenize_raises_when_preprocess_command_fails(monkeypatch, tmp_path):
    downloader = DummyDownloader(
        data_dir=str(tmp_path),
        tokenizer_type="CharLevelTokenizer",
        num_workers=1,
    )
    monkeypatch.setattr(corpora.os, "system", lambda cmd: 1)

    with pytest.raises(RuntimeError, match="Failed to tokenize dataset dummy"):
        downloader.tokenize()
