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

from typing import NamedTuple

import torch


class PackedSequenceModelInputs(NamedTuple):
    tokens: torch.Tensor
    position_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    num_documents: torch.Tensor
    max_seqlen: torch.Tensor
    attention_mask: torch.Tensor


class PackedSequenceBatch(NamedTuple):
    tokens: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    num_documents: torch.Tensor
    max_seqlen: torch.Tensor

    def model_inputs(self):
        return PackedSequenceModelInputs(
            self.tokens,
            self.position_ids,
            self.cu_seqlens,
            self.num_documents,
            self.max_seqlen,
            self.attention_mask,
        )

    def loss_inputs(self):
        return self.labels, self.loss_mask


def normalize_packed_sequence_batch(
    tokens, labels, label_mask, cu_seqlens, max_seqlen, eod_token, eod_mask_loss
):
    """Flatten a [B, S] microbatch to [1, B * S] with one cu_seqlens for all documents."""
    num_tokens = tokens.numel()
    lengths = (cu_seqlens[:, 1:] - cu_seqlens[:, :-1]).reshape(-1)
    lengths = lengths[lengths > 0]
    starts = torch.cat([lengths.new_zeros(1), lengths.cumsum(0, dtype=lengths.dtype)])
    # Pipeline stages need a fixed shape, so cu_seqlens is padded to B * S + 1 entries.
    padded_cu_seqlens = torch.cat(
        [starts, starts[-1:].expand(num_tokens + 1 - starts.numel())]
    )
    position_ids = torch.arange(
        num_tokens, device=tokens.device
    ) - torch.repeat_interleave(starts[:-1].long(), lengths, output_size=num_tokens)
    loss_mask = label_mask.float()
    if eod_mask_loss:
        if eod_token is None:
            raise ValueError(
                "eod_mask_loss is True but the tokenizer has no "
                "end-of-document token ('<|endoftext|>' is not in the "
                "vocab). Set 'tokenizer_eod_token' to the name of the "
                "end-of-document token or disable eod_mask_loss."
            )
        loss_mask = loss_mask * (tokens != eod_token)
    return PackedSequenceBatch(
        tokens=tokens.reshape(1, -1),
        labels=labels.reshape(1, -1),
        loss_mask=loss_mask.reshape(1, -1),
        attention_mask=tokens.new_zeros(1, dtype=torch.bool),
        position_ids=position_ids.unsqueeze(0),
        cu_seqlens=padded_cu_seqlens,
        num_documents=starts.new_tensor(starts.numel() - 1),
        max_seqlen=max_seqlen.max(),
    )
