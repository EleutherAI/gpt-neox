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

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from megatron.model.fused_softmax import softcap_attention_logits
from megatron.model.transformer import ParallelSelfAttention
from megatron.neox_arguments import NeoXArgs
from megatron.neox_arguments import arguments as neox_arguments
from tests.common import BASE_CONFIG


@pytest.mark.cpu
def test_softcap_attention_logits_matches_paper_formula():
    logits = torch.tensor([-100.0, -10.0, 0.0, 10.0, 100.0], requires_grad=True)
    cap = 50.0

    actual = softcap_attention_logits(logits, cap)
    expected = cap * torch.tanh(logits / cap)

    torch.testing.assert_close(actual, expected)
    assert actual.abs().max() < cap

    actual.sum().backward()
    torch.testing.assert_close(logits.grad, 1 - torch.tanh(logits.detach() / cap) ** 2)


@pytest.mark.cpu
def test_softcap_preserves_query_key_layer_scaling_cap():
    logits = torch.tensor([-80.0, -5.0, 0.0, 20.0, 90.0])
    cap = 50.0
    layer_coefficient = 4

    pre_softmax = softcap_attention_logits(
        logits / layer_coefficient, cap / layer_coefficient
    )

    torch.testing.assert_close(
        pre_softmax * layer_coefficient,
        softcap_attention_logits(logits, cap),
    )


@pytest.mark.cpu
@pytest.mark.parametrize("training", [True, False])
def test_flash_attention_receives_softcap(training):
    class FlashRecorder:
        def __init__(self):
            self.kwargs = None

        def __call__(self, query, *args, **kwargs):
            self.kwargs = kwargs
            return torch.zeros_like(query)

    flash = FlashRecorder()
    attention = SimpleNamespace(
        use_flash_attention=True,
        use_triton=False,
        num_kv_heads_per_partition=4,
        sliding_window_width=None,
        pos_emb="rotary",
        attn_logit_softcapping=50.0,
        training=training,
        dropout_p=0.0,
        flash_qkv_fn=flash,
        flash_varlen_qkv_fn=flash,
    )
    query = torch.randn(3, 2, 4, 8)

    output = ParallelSelfAttention.flash_attention(
        attention,
        query,
        torch.randn_like(query),
        torch.randn_like(query),
    )

    assert flash.kwargs["softcap"] == 50.0
    assert output.shape == (2, 4, 3, 8)


def _make_neox_args(**updates):
    config = deepcopy(BASE_CONFIG)
    config["global_num_gpus"] = 1
    config.update(updates)
    return NeoXArgs.from_dict(config)


@pytest.mark.cpu
def test_attention_softcapping_config_accepts_positive_cap():
    neox_args = _make_neox_args(attn_logit_softcapping=50.0)
    assert neox_args.attn_logit_softcapping == 50.0


@pytest.mark.cpu
@pytest.mark.parametrize("cap", [0.0, -1.0])
def test_attention_softcapping_config_rejects_nonpositive_cap(cap):
    with pytest.raises(ValueError, match="must be greater than 0"):
        _make_neox_args(attn_logit_softcapping=cap)


@pytest.mark.cpu
def test_attention_softcapping_config_rejects_unsupported_backends():
    with pytest.raises(ValueError, match="not supported with `te_mha`"):
        _make_neox_args(attn_logit_softcapping=50.0, te_mha=True)

    with pytest.raises(ValueError, match="not supported by sparse attention"):
        _make_neox_args(
            attn_logit_softcapping=50.0,
            attention_config=[[["local"], "all"]],
        )


@pytest.mark.cpu
def test_attention_softcapping_requires_flash_attention_2_6(monkeypatch):
    monkeypatch.setattr(neox_arguments, "version", lambda _: "2.5.6")

    with pytest.raises(ValueError, match="flash-attn >= 2.6.0"):
        _make_neox_args(
            attn_logit_softcapping=50.0,
            attention_config=[[["flash"], "all"]],
        )
