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

from types import SimpleNamespace

import pytest
import torch

from megatron import training
from megatron.data.packed_sequence import (
    PackedSequenceBatch,
    PackedSequenceModelInputs,
)


def _neox_args(enabled=True, seq_length=4):
    return SimpleNamespace(
        inter_document_attention_masking=enabled,
        seq_length=seq_length,
        tokenizer=SimpleNamespace(eod=-1),
        eod_mask_loss=False,
        sliding_window_width=None,
        train_impl="normal",
        train_label_data_paths=None,
        is_pipe_parallel=False,
        memory_profiling=False,
        iteration=0,
        curriculum_learning=False,
        fp16_lm_cross_entropy=False,
    )


def _batch():
    return {
        "text": torch.arange(10, dtype=torch.int64).view(2, 5),
        "cu_seqlens": torch.tensor(
            [[0, 2, 4, 4, 4], [0, 4, 4, 4, 4]], dtype=torch.int32
        ),
        "max_seqlen": torch.tensor([2, 4], dtype=torch.int32),
    }


@pytest.fixture
def local_broadcast(monkeypatch):
    calls = []

    def broadcast_data(keys, source, datatype):
        calls.append((keys, datatype))
        return {key: source[key] for key in keys}

    monkeypatch.setattr(training.mpu, "broadcast_data", broadcast_data)
    return calls


@pytest.fixture
def packed_batch_result():
    return PackedSequenceBatch(
        tokens=torch.tensor([[10, 11]]),
        labels=torch.tensor([[11, 12]]),
        loss_mask=torch.ones((1, 2)),
        attention_mask=torch.zeros(1, dtype=torch.bool),
        position_ids=torch.tensor([[0, 1]]),
        cu_seqlens=torch.tensor([0, 2, 2], dtype=torch.int32),
        num_documents=torch.tensor(1, dtype=torch.int32),
        max_seqlen=torch.tensor(2, dtype=torch.int32),
    )


@pytest.mark.cpu
def test_get_batch_packed_path_skips_dense_mask_and_preserves_loss_masking(
    monkeypatch, local_broadcast
):
    data = _batch()
    data["label"] = data["text"].clone()
    data["label"][0, 2] = -1
    neox_args = _neox_args()
    neox_args.tokenizer.eod = 2
    neox_args.eod_mask_loss = True

    monkeypatch.setattr(
        training,
        "get_ltor_masks_and_position_ids",
        lambda *args, **kwargs: pytest.fail("dense mask helper was called"),
    )

    batch = training._get_batch(
        neox_args=neox_args,
        tokenizer=neox_args.tokenizer,
        keys=["text", "label"],
        data=data,
        datatype=torch.int64,
    )

    assert local_broadcast == [
        (["text", "label"], torch.int64),
        (["cu_seqlens", "max_seqlen"], torch.int32),
    ]
    assert isinstance(batch, PackedSequenceBatch)
    tokens = batch.tokens
    labels = batch.labels
    loss_mask = batch.loss_mask
    attention_mask = batch.attention_mask
    position_ids = batch.position_ids
    assert torch.equal(tokens, torch.tensor([[0, 1, 2, 3, 5, 6, 7, 8]]))
    assert attention_mask.shape == (1,)
    assert attention_mask.dtype == torch.bool
    assert not attention_mask.item()
    assert torch.equal(
        labels, torch.tensor([[1, 0, 3, 4, 6, 7, 8, 9]])
    )
    assert torch.equal(
        loss_mask, torch.tensor([[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    )
    assert torch.equal(
        position_ids, torch.tensor([[0, 1, 0, 1, 0, 1, 2, 3]])
    )
    assert torch.equal(
        batch.cu_seqlens,
        torch.tensor([0, 2, 4, 8, 8, 8, 8, 8, 8], dtype=torch.int32),
    )
    assert batch.num_documents == 3
    assert batch.max_seqlen == 4


@pytest.mark.cpu
def test_get_batch_preserves_exact_behavior_when_disabled(local_broadcast):
    data = _batch()
    neox_args = _neox_args(enabled=False)
    neox_args.tokenizer.eod = 2
    neox_args.eod_mask_loss = True

    tokens, labels, loss_mask, attention_mask, position_ids = training._get_batch(
        neox_args=neox_args,
        tokenizer=neox_args.tokenizer,
        keys=["text"],
        data=data,
        datatype=torch.int64,
    )

    assert local_broadcast == [(["text"], torch.int64)]
    assert torch.equal(tokens, torch.tensor([[0, 1, 2, 3], [5, 6, 7, 8]]))
    assert torch.equal(labels, torch.tensor([[1, 2, 3, 4], [6, 7, 8, 9]]))
    assert torch.equal(
        loss_mask,
        torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]),
    )
    assert torch.equal(
        position_ids,
        torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
    )
    assert torch.equal(
        attention_mask,
        torch.tensor(
            [
                [
                    [
                        [False, True, True, True],
                        [False, False, True, True],
                        [False, False, False, True],
                        [False, False, False, False],
                    ]
                ]
            ]
        ),
    )


@pytest.mark.cpu
def test_get_batch_sequential_preserves_packed_mask_sentinel(monkeypatch):
    forward_input = PackedSequenceModelInputs(
        tokens=torch.tensor([[0, 1, 2, 3]]),
        position_ids=torch.tensor([[0, 1, 0, 1]]),
        cu_seqlens=torch.tensor([0, 2, 4, 4, 4], dtype=torch.int32),
        num_documents=torch.tensor(2, dtype=torch.int32),
        max_seqlen=torch.tensor(2, dtype=torch.int32),
        attention_mask=torch.zeros(1, dtype=torch.bool),
    )

    monkeypatch.setattr(
        training,
        "get_ltor_masks_and_position_ids",
        lambda *args, **kwargs: pytest.fail("dense mask helper was called"),
    )

    result = training.get_batch_sequential(forward_input, _neox_args())

    assert result is forward_input


@pytest.mark.cpu
def test_get_batch_sequential_accepts_ordinary_input_when_feature_is_enabled(
    monkeypatch,
):
    tokens = torch.tensor([[0, 1, 2, 3]])
    position_ids = torch.tensor([[0, 1, 2, 3]])
    dense_attention_mask = torch.zeros((1, 1, 4, 4), dtype=torch.bool)

    def get_masks(data, eod_token, eod_mask_loss):
        assert data is tokens
        return dense_attention_mask, None, position_ids

    monkeypatch.setattr(training, "get_ltor_masks_and_position_ids", get_masks)

    forward_input = (tokens, position_ids, dense_attention_mask)
    result = training.get_batch_sequential(forward_input, _neox_args(enabled=True))

    assert result[0] is tokens
    assert result[1] is position_ids
    assert result[2] is dense_attention_mask


@pytest.mark.cpu
def test_get_batch_pipe_keeps_metadata_in_model_inputs(
    monkeypatch, packed_batch_result
):
    monkeypatch.setattr(
        training, "_get_batch", lambda *args, **kwargs: packed_batch_result
    )

    model_inputs, loss_inputs = training.get_batch_pipe(
        data=None, neox_args=_neox_args()
    )

    assert isinstance(model_inputs, PackedSequenceModelInputs)
    assert all(
        actual is expected
        for actual, expected in zip(
            model_inputs, packed_batch_result.model_inputs()
        )
    )
    assert all(
        actual is expected
        for actual, expected in zip(loss_inputs, packed_batch_result.loss_inputs())
    )


@pytest.mark.cpu
@pytest.mark.parametrize("packed", [False, True], ids=["ordinary", "packed"])
def test_forward_step_builds_model_inputs_and_keeps_loss_inputs_separate(
    monkeypatch, packed_batch_result, packed
):
    neox_args = _neox_args()
    observed = {}
    if packed:
        batch_result = packed_batch_result
        expected_model_inputs = packed_batch_result.model_inputs()
        expected_loss_inputs = packed_batch_result.loss_inputs()
    else:
        dense_attention_mask = torch.zeros((1, 1, 2, 2), dtype=torch.bool)
        batch_result = (
            packed_batch_result.tokens,
            packed_batch_result.labels,
            packed_batch_result.loss_mask,
            dense_attention_mask,
            packed_batch_result.position_ids,
        )
        expected_model_inputs = (
            packed_batch_result.tokens,
            packed_batch_result.position_ids,
            dense_attention_mask,
        )
        expected_loss_inputs = packed_batch_result.loss_inputs()

    monkeypatch.setattr(
        training,
        "get_batch",
        lambda neox_args, data_iterator: batch_result,
    )

    def model(model_inputs, neox_args):
        observed["model_inputs"] = model_inputs
        return torch.tensor(0.0)

    def cross_entropy(output, loss_inputs, _fp16):
        observed["loss_inputs"] = loss_inputs
        return torch.tensor(1.5)

    monkeypatch.setattr(training, "cross_entropy", cross_entropy)

    loss, metrics = training.forward_step(
        data_iterator=None,
        model=model,
        neox_args=neox_args,
        timers=None,
    )

    assert all(
        actual is expected
        for actual, expected in zip(
            observed["model_inputs"], expected_model_inputs
        )
    )
    assert all(
        actual is expected
        for actual, expected in zip(observed["loss_inputs"], expected_loss_inputs)
    )
    assert loss == 1.5
    assert metrics == {}


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        (
            "cu_seqlens",
            torch.zeros((2, 4), dtype=torch.int32),
            "expects cu_seqlens with shape",
        ),
        (
            "max_seqlen",
            torch.zeros((2, 1), dtype=torch.int32),
            "expects max_seqlen with shape",
        ),
        (
            "cu_seqlens",
            torch.zeros((2, 5), dtype=torch.int64),
            "expects cu_seqlens to be int32",
        ),
        (
            "max_seqlen",
            torch.zeros(2, dtype=torch.int64),
            "expects max_seqlen to be int32",
        ),
    ],
)
def test_packed_metadata_shape_validation(monkeypatch, field, value, error):
    data = _batch()
    data[field] = value

    monkeypatch.setattr(
        training.mpu,
        "broadcast_data",
        lambda keys, source, datatype: {key: source[key] for key in keys},
    )

    with pytest.raises(AssertionError, match=error):
        training._broadcast_packed_sequence_metadata(
            _neox_args(), data, data["text"]
        )
