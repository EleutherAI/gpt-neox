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

"""Pure tensor transformations for packed-sequence microbatches."""

from typing import NamedTuple

import torch


class PackedSequenceModelInputs(NamedTuple):
    """Structured view of the flat packed input accepted by the model."""

    tokens: torch.Tensor
    position_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    num_documents: torch.Tensor
    max_seqlen: torch.Tensor
    attention_mask: torch.Tensor


class PackedSequenceBatch(NamedTuple):
    """Normalized tensors and metadata for one packed microbatch."""

    tokens: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    num_documents: torch.Tensor
    max_seqlen: torch.Tensor

    def model_inputs(self):
        """Return the flat tensor-only input consumed by the model."""
        return PackedSequenceModelInputs(
            tokens=self.tokens,
            position_ids=self.position_ids,
            cu_seqlens=self.cu_seqlens,
            num_documents=self.num_documents,
            max_seqlen=self.max_seqlen,
            attention_mask=self.attention_mask,
        )

    def loss_inputs(self):
        """Return labels and loss weights kept outside the model context."""
        return self.labels, self.loss_mask


def _get_document_lengths(cu_seqlens):
    """Recover real document lengths from fixed-width collated boundaries.

    For example, two samples of length six may be collated as::

        cu_seqlens = [
            [0, 1, 3, 6, 6, 6, 6],
            [0, 4, 6, 6, 6, 6, 6],
        ]

    Adjacent differences produce ``[[1, 2, 3, 0, 0, 0],
    [4, 2, 0, 0, 0, 0]]``. Flattening in row-major order and removing the
    zero-length collation padding returns ``[1, 2, 3, 4, 2]``: all document
    lengths from sample zero followed by all document lengths from sample one.

    Args:
        cu_seqlens: An int32 tensor with shape ``[B, S + 1]``.

    Returns:
        A contiguous int32 tensor containing the positive document lengths in
        microbatch token order.
    """
    document_lengths_by_sample = cu_seqlens[:, 1:] - cu_seqlens[:, :-1]

    # The dataset pads each row by repeating its terminal offset. Adjacent
    # differences turn those entries into zeros. Boolean indexing removes them
    # while preserving the row-major order of documents across the microbatch.
    document_lengths = document_lengths_by_sample.reshape(-1)
    return document_lengths[document_lengths > 0].contiguous()


def _merge_metadata(document_lengths, max_seqlen):
    """Merge ordered document lengths into microbatch-wide packed metadata.

    For document lengths ``[1, 2, 3, 4, 2]`` from two samples of length six,
    the cumulative result is ``[0, 1, 3, 6, 10, 12]``. The sample join at
    offset six appears only once. Per-sample maximums such as ``[3, 4]`` are
    reduced to the scalar microbatch maximum ``4``.

    Args:
        document_lengths: Positive int32 lengths in microbatch token order.
        max_seqlen: Per-sample int32 maximums with shape ``[B]``.

    Returns:
        The unpadded merged cumulative boundaries and scalar maximum length.
    """
    merged_cu_seqlens = torch.cat(
        [
            document_lengths.new_zeros(1),
            document_lengths.cumsum(dim=0, dtype=document_lengths.dtype),
        ]
    )
    return merged_cu_seqlens, max_seqlen.max()


def _pad_metadata(merged_cu_seqlens, batch_size, sequence_length):
    """Pad merged boundaries to a fixed shape for pipeline transport.

    DeepSpeed caches pipeline activation shapes when dynamic shapes are
    disabled. The real number of document boundaries varies by microbatch, so
    repeat the terminal boundary until every microbatch has the maximum shape
    ``[B * S + 1]``. ``num_documents`` records how much of the tensor is real.

    For ``B = 2`` and ``S = 6``, merged boundaries
    ``[0, 1, 3, 6, 10, 12]`` become::

        [0, 1, 3, 6, 10, 12, 12, 12, 12, 12, 12, 12, 12]

    and ``num_documents`` is the scalar ``5``.

    Args:
        merged_cu_seqlens: Unpadded int32 boundaries with shape ``[D + 1]``.
        batch_size: Runtime microbatch size ``B``.
        sequence_length: Per-sample input-token length ``S``.

    Returns:
        Fixed-width int32 boundaries and a scalar int32 document count.
    """
    transport_length = batch_size * sequence_length + 1
    padding_length = transport_length - merged_cu_seqlens.numel()

    padded_cu_seqlens = torch.cat(
        [merged_cu_seqlens, merged_cu_seqlens[-1:].expand(padding_length)]
    )
    num_documents = merged_cu_seqlens.new_tensor(
        merged_cu_seqlens.numel() - 1
    )
    return padded_cu_seqlens, num_documents


def _flatten_aligned_tensors(tokens, labels, loss_mask, position_ids):
    """Flatten sequence-aligned tensors into microbatch packed-token order.

    Reshaping ``[[t00, t01], [t10, t11]]`` produces
    ``[[t00, t01, t10, t11]]``. Applying the same row-major transformation to
    inputs, targets, loss weights, and positions keeps every token aligned with
    its corresponding training metadata.

    Args:
        tokens: Input token IDs with shape ``[B, S]``.
        labels: Target token IDs with shape ``[B, S]``.
        loss_mask: Per-token loss weights with shape ``[B, S]``.
        position_ids: Per-token positions with shape ``[B, S]``.

    Returns:
        The four contiguous tensors, each with shape ``[1, B * S]``.
    """
    return tuple(
        tensor.reshape(1, -1).contiguous()
        for tensor in (tokens, labels, loss_mask, position_ids)
    )


def _get_position_ids(document_lengths, total_tokens):
    """Build position IDs that restart at every packed document boundary.

    Explanation of working:

    Consider five documents with a total of twelve tokens::

        document_lengths = [1, 2, 3, 4, 2]

    Cumulative sums of all but the final length, with an initial zero, give
    each document's start in the flattened token stream::

        document_starts = [0, 1, 3, 6, 10]

    Repeating each start by its corresponding document length associates every
    token with the start of the document containing it::

        token_document_starts = [0, 1, 1, 3, 3, 3, 6, 6, 6, 6, 10, 10]

    The global offsets for the twelve packed tokens are::

        token_offsets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    Subtracting the document start from each global offset produces positions
    that restart from zero at every document boundary::

        position_ids = [[0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1]]

    Args:
        document_lengths: Positive int32 lengths in packed-token order.
        total_tokens: Expected number of tokens across the microbatch.

    Returns:
        Contiguous int64 position IDs with shape ``[1, total_tokens]``.
    """
    document_starts = torch.cat(
        [
            document_lengths.new_zeros(1, dtype=torch.long),
            document_lengths[:-1].cumsum(dim=0, dtype=torch.long),
        ]
    )
    token_document_starts = torch.repeat_interleave(
        document_starts,
        document_lengths,
        output_size=total_tokens,
    )
    token_offsets = torch.arange(
        total_tokens, dtype=torch.long, device=document_lengths.device
    )
    return (token_offsets - token_document_starts).unsqueeze(0).contiguous()


def _get_masks(tokens, label_mask, eod_token, eod_mask_loss):
    """Build packed loss weights without allocating a dense attention mask.

    Packed attention derives causality and document isolation from
    ``cu_seqlens``, so the normal ``[1, 1, S, S]`` boolean mask is unnecessary.
    A fixed one-element boolean tensor acts as the tensor-only placeholder that
    DeepSpeed can carry through pipeline communication and checkpointing.

    The loss mask preserves the normal batch path: every token starts with
    weight one, EOD positions are set to zero when ``eod_mask_loss`` is
    enabled, and the result is combined with the label validity mask.

    Returns:
        The boolean attention-mask sentinel and float32 per-token loss mask.
    """
    attention_mask = tokens.new_zeros(1, dtype=torch.bool)
    loss_mask = torch.ones(tokens.size(), dtype=torch.float, device=tokens.device)
    if eod_mask_loss:
        loss_mask[tokens == eod_token] = 0.0
    return attention_mask, label_mask.to(loss_mask.dtype) * loss_mask


def normalize_packed_sequence_batch(
    tokens,
    labels,
    label_mask,
    cu_seqlens,
    max_seqlen,
    eod_token,
    eod_mask_loss,
):
    """Normalize a collated microbatch into packed-token order.

    The input token-aligned tensors have shape ``[B, S]``. Dataset metadata is
    collated as fixed-width ``cu_seqlens: [B, S + 1]`` and per-sample
    ``max_seqlen: [B]``. The result contains flattened token-aligned tensors
    with shape ``[1, B * S]``, fixed-width microbatch boundaries with shape
    ``[B * S + 1]``, scalar document-count and maximum-length metadata, and a
    one-element attention-mask sentinel.

    This function performs only deterministic tensor transformations. Data
    broadcasting and feature selection remain responsibilities of the caller.
    """
    batch_size, sequence_length = tokens.shape
    document_lengths = _get_document_lengths(cu_seqlens)
    merged_cu_seqlens, max_seqlen = _merge_metadata(
        document_lengths, max_seqlen
    )
    cu_seqlens, num_documents = _pad_metadata(
        merged_cu_seqlens,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    attention_mask, loss_mask = _get_masks(
        tokens=tokens,
        label_mask=label_mask,
        eod_token=eod_token,
        eod_mask_loss=eod_mask_loss,
    )
    position_ids = _get_position_ids(
        document_lengths, total_tokens=tokens.numel()
    ).view_as(tokens)
    tokens, labels, loss_mask, position_ids = _flatten_aligned_tensors(
        tokens, labels, loss_mask, position_ids
    )

    return PackedSequenceBatch(
        tokens=tokens,
        labels=labels,
        loss_mask=loss_mask,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cu_seqlens=cu_seqlens,
        num_documents=num_documents,
        max_seqlen=max_seqlen,
    )
