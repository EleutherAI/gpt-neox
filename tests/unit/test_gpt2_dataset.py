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

import numpy as np
import pytest

from megatron.data.gpt2_dataset import GPT2Dataset


class _IndexedDataset:
    def __init__(self, documents):
        self.documents = [
            np.asarray(document, dtype=np.int64) for document in documents
        ]
        self.sizes = np.asarray(
            [len(document) for document in documents], dtype=np.int32
        )

    def get(self, idx, offset=0, length=None):
        document = self.documents[idx]
        if length is None:
            return document[offset:]
        return document[offset : offset + length]


def _build_dataset(documents, sample_idx, seq_length):
    indexed_dataset = _IndexedDataset(documents)
    dataset = GPT2Dataset(
        name="test",
        data_prefix="unused",
        documents=np.arange(len(documents), dtype=np.int32),
        indexed_dataset=indexed_dataset,
        num_samples=1,
        num_epochs=1,
        seq_length=seq_length,
        seed=1234,
        build_index_mappings=False,
        inter_document_attention_masking=True,
    )
    dataset.doc_idx = np.arange(len(documents), dtype=np.int32)
    dataset.sample_idx = np.asarray(sample_idx, dtype=np.int64)
    dataset.shuffle_idx = np.asarray([0], dtype=np.int64)
    return dataset


@pytest.mark.cpu
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            {
                "documents": [np.arange(5), np.arange(10, 16)],
                "sample_idx": [[0, 2], [1, 5]],
                "expected_text": [2, 3, 4, 10, 11, 12, 13, 14, 15],
                "expected_cu_seqlens": [0, 3, 8, 8, 8, 8, 8, 8, 8],
                "expected_max_seqlen": 5,
            },
            id="partial-fragments",
        ),
        pytest.param(
            {
                "documents": [np.arange(20)],
                "sample_idx": [[0, 4], [0, 12]],
                "expected_text": np.arange(4, 13),
                "expected_cu_seqlens": [0, 8, 8, 8, 8, 8, 8, 8, 8],
                "expected_max_seqlen": 8,
            },
            id="single-partial-document",
        ),
        pytest.param(
            {
                "documents": [np.arange(8), np.arange(10, 11)],
                "sample_idx": [[0, 0], [1, 0]],
                "expected_text": [0, 1, 2, 3, 4, 5, 6, 7, 10],
                "expected_cu_seqlens": [0, 8, 8, 8, 8, 8, 8, 8, 8],
                "expected_max_seqlen": 8,
            },
            id="label-only-final-fragment",
        ),
        pytest.param(
            {
                "documents": [np.arange(3), np.arange(10, 12)],
                "sample_idx": [[0, 0], [1, 1]],
                "expected_text": [0, 1, 2, 10, 11, 0, 0, 0, 0],
                "expected_cu_seqlens": [0, 3, 8, 8, 8, 8, 8, 8, 8],
                "expected_max_seqlen": 5,
            },
            id="sample-padding",
        ),
        pytest.param(
            {
                "documents": [np.arange(3), np.arange(10, 11)],
                "sample_idx": [[0, 0], [1, 0]],
                "expected_text": [0, 1, 2, 10, 0, 0, 0, 0, 0],
                "expected_cu_seqlens": [0, 3, 8, 8, 8, 8, 8, 8, 8],
                "expected_max_seqlen": 5,
            },
            id="sample-padding-after-single-token-document",
        ),
    ],
)
def test_inter_document_metadata_edge_cases(case):
    dataset = _build_dataset(
        documents=case["documents"],
        sample_idx=case["sample_idx"],
        seq_length=8,
    )

    sample = dataset[0]

    np.testing.assert_array_equal(sample["text"], case["expected_text"])
    np.testing.assert_array_equal(sample["cu_seqlens"], case["expected_cu_seqlens"])
    assert sample["cu_seqlens"].dtype == np.int32
    assert sample["max_seqlen"] == np.int32(case["expected_max_seqlen"])
