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

import pytest
import torch

from megatron.data.packed_sequence import normalize_packed_sequence_batch


@pytest.mark.cpu
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            {
                "tokens": [[10, 11, 12, 13]],
                "label_mask": [[True, True, False, True]],
                "cu_seqlens": [[0, 2, 4, 4, 4]],
                "max_seqlen": [2],
                "eod_token": 11,
                "expected_loss_mask": [[1.0, 0.0, 0.0, 1.0]],
                "expected_position_ids": [[0, 1, 0, 1]],
                "expected_cu_seqlens": [0, 2, 4, 4, 4],
                "expected_num_documents": 2,
                "expected_max_seqlen": 2,
            },
            id="single-sample",
        ),
        pytest.param(
            {
                "tokens": [[0, 1, 2, 3], [4, 5, 6, 7]],
                "label_mask": [
                    [True, True, False, True],
                    [True, True, True, False],
                ],
                "cu_seqlens": [
                    [0, 1, 4, 4, 4],
                    [0, 2, 3, 4, 4],
                ],
                "max_seqlen": [3, 2],
                "eod_token": 5,
                "expected_loss_mask": [
                    [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
                ],
                "expected_position_ids": [[0, 0, 1, 2, 0, 1, 0, 0]],
                "expected_cu_seqlens": [0, 1, 4, 6, 7, 8, 8, 8, 8],
                "expected_num_documents": 5,
                "expected_max_seqlen": 3,
            },
            id="multi-sample-different-document-counts",
        ),
    ],
)
def test_normalize_packed_sequence_batch(case):
    tokens = torch.tensor(case["tokens"], dtype=torch.int64)
    labels = tokens + 100
    label_mask = torch.tensor(case["label_mask"], dtype=torch.bool)
    batch_size, sequence_length = tokens.shape

    result = normalize_packed_sequence_batch(
        tokens=tokens,
        labels=labels,
        label_mask=label_mask,
        cu_seqlens=torch.tensor(case["cu_seqlens"], dtype=torch.int32),
        max_seqlen=torch.tensor(case["max_seqlen"], dtype=torch.int32),
        eod_token=case["eod_token"],
        eod_mask_loss=True,
    )

    assert torch.equal(result.tokens, tokens.reshape(1, -1))
    assert torch.equal(result.labels, labels.reshape(1, -1))
    assert torch.equal(
        result.loss_mask, torch.tensor(case["expected_loss_mask"])
    )
    assert torch.equal(
        result.position_ids,
        torch.tensor(case["expected_position_ids"], dtype=torch.int64),
    )
    assert torch.equal(
        result.cu_seqlens,
        torch.tensor(case["expected_cu_seqlens"], dtype=torch.int32),
    )
    assert result.num_documents.item() == case["expected_num_documents"]
    assert result.max_seqlen.item() == case["expected_max_seqlen"]

    total_tokens = batch_size * sequence_length
    assert result.tokens.shape == result.labels.shape == (1, total_tokens)
    assert result.loss_mask.shape == result.position_ids.shape == (1, total_tokens)
    assert result.cu_seqlens.shape == (total_tokens + 1,)
    assert result.num_documents.shape == result.max_seqlen.shape == ()
    assert result.cu_seqlens.dtype == torch.int32
    assert result.num_documents.dtype == result.max_seqlen.dtype == torch.int32
    assert result.attention_mask.shape == (1,)
    assert result.attention_mask.dtype == torch.bool
    assert not result.attention_mask.item()
    assert all(
        tensor.is_contiguous()
        for tensor in (
            result.tokens,
            result.labels,
            result.loss_mask,
            result.position_ids,
            result.cu_seqlens,
        )
    )

    real_boundaries = result.cu_seqlens[: result.num_documents + 1]
    document_lengths = real_boundaries[1:] - real_boundaries[:-1]
    assert real_boundaries[-1].item() == total_tokens
    assert torch.all(document_lengths > 0)
    assert result.max_seqlen == document_lengths.max()
    if batch_size > 1:
        # A sample join is one boundary, not the terminal boundary from both
        # adjacent samples.
        assert torch.count_nonzero(real_boundaries == sequence_length).item() == 1


@pytest.mark.cpu
def test_normalized_metadata_transport_shape_is_independent_of_document_count():
    tokens = torch.arange(8, dtype=torch.int64).view(2, 4)
    variants = [
        (
            torch.tensor(
                [[0, 4, 4, 4, 4], [0, 4, 4, 4, 4]], dtype=torch.int32
            ),
            torch.tensor([4, 4], dtype=torch.int32),
        ),
        (
            torch.tensor(
                [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=torch.int32
            ),
            torch.tensor([1, 1], dtype=torch.int32),
        ),
    ]

    results = [
        normalize_packed_sequence_batch(
            tokens=tokens,
            labels=tokens + 1,
            label_mask=torch.ones_like(tokens, dtype=torch.bool),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            eod_token=-1,
            eod_mask_loss=False,
        )
        for cu_seqlens, max_seqlen in variants
    ]

    assert results[0].cu_seqlens.shape == results[1].cu_seqlens.shape == (9,)
    assert [result.num_documents.item() for result in results] == [2, 8]
    assert torch.equal(
        results[0].cu_seqlens,
        torch.tensor([0, 4, 8, 8, 8, 8, 8, 8, 8], dtype=torch.int32),
    )
    assert torch.equal(results[1].cu_seqlens, torch.arange(9, dtype=torch.int32))
