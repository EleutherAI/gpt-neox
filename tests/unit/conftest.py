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

import pytest


@pytest.fixture
def write_hf_tokenizer(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    def write(vocab):
        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        path = tmp_path / "tokenizer.json"
        tokenizer.save(str(path))
        return str(path)

    return write


@pytest.fixture
def gemma_like_file(write_hf_tokenizer):
    # Laid out like Gemma's special tokens: no <|endoftext|>, EOS is <eos>
    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "hello": 4, "world": 5}
    return write_hf_tokenizer(vocab)


@pytest.fixture
def neox_like_file(write_hf_tokenizer):
    vocab = {"<|endoftext|>": 0, "<|padding|>": 1, "hello": 2, "world": 3, "<unk>": 4}
    return write_hf_tokenizer(vocab)


@pytest.fixture
def spm_file(tmp_path):
    import sentencepiece as spm

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
