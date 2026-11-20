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

import pytest
import torch

from megatron.model.fused_softmax import (
    FusedScaleMaskSoftmax,
    SoftmaxFusionTypes,
    softclip_attention_probs,
)
from megatron.neox_arguments import NeoXArgs
from megatron.neox_arguments import arguments as neox_arguments
from tests.common import BASE_CONFIG


def _make_softmax(soft_clip, zeta=1.03, gamma=-0.03):
    return FusedScaleMaskSoftmax(
        input_in_fp16=False,
        input_in_bf16=False,
        fusion_type=SoftmaxFusionTypes.none,
        mask_func=lambda scores, mask: scores.masked_fill(mask, -10000.0),
        softmax_in_fp32=False,
        scale=None,
        soft_clip=soft_clip,
        soft_clip_zeta=zeta,
        soft_clip_gamma=gamma,
    )


@pytest.mark.cpu
def test_softclip_attention_probs_matches_paper_formula():
    probs = torch.tensor([0.01, 0.25, 0.5, 0.99], requires_grad=True)

    actual = softclip_attention_probs(probs, zeta=1.03, gamma=-0.03)
    expected = torch.clamp(1.06 * probs - 0.03, min=0.0, max=1.0)

    torch.testing.assert_close(actual, expected)
    assert actual[0] == 0.0
    assert actual[-1] == 1.0

    actual.sum().backward()
    torch.testing.assert_close(probs.grad, torch.tensor([0.0, 1.06, 1.06, 0.0]))


@pytest.mark.cpu
def test_fused_scale_mask_softmax_applies_softclip_after_softmax():
    logits = torch.tensor([[[[5.0, 0.0, -5.0]]]])
    zeta = 1.1
    gamma = -0.2

    actual = _make_softmax(soft_clip=True, zeta=zeta, gamma=gamma)(logits, mask=None)
    expected = softclip_attention_probs(
        torch.softmax(logits, dim=-1), zeta=zeta, gamma=gamma
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.cpu
def test_fused_scale_mask_softmax_is_unchanged_by_default():
    logits = torch.tensor([[[[2.0, 1.0, -1.0]]]])

    actual = _make_softmax(soft_clip=False, zeta=2.0, gamma=-1.0)(logits, mask=None)

    torch.testing.assert_close(actual, torch.softmax(logits, dim=-1))


def _make_neox_args(**updates):
    config = deepcopy(BASE_CONFIG)
    config["global_num_gpus"] = 1
    config.update(updates)
    return NeoXArgs.from_dict(config)


@pytest.mark.cpu
def test_attention_softmax_clipping_uses_paper_defaults():
    neox_args = _make_neox_args(attn_softmax_clipping=True)

    assert neox_args.attn_softmax_clipping_zeta == 1.03
    assert neox_args.attn_softmax_clipping_gamma == -0.03


@pytest.mark.cpu
def test_attention_softmax_clipping_accepts_custom_bounds():
    neox_args = _make_neox_args(
        attn_softmax_clipping=True,
        attn_softmax_clipping_zeta=1.1,
        attn_softmax_clipping_gamma=-0.1,
    )
    assert neox_args.attn_softmax_clipping is True
    assert neox_args.attn_softmax_clipping_zeta == 1.1
    assert neox_args.attn_softmax_clipping_gamma == -0.1


@pytest.mark.cpu
@pytest.mark.parametrize(
    "updates, expected_error",
    [
        ({"attn_softmax_clipping_zeta": 0.99}, "zeta.*greater than or equal"),
        ({"attn_softmax_clipping_gamma": 0.01}, "gamma.*less than or equal"),
    ],
)
def test_attention_softmax_clipping_rejects_invalid_bounds(updates, expected_error):
    with pytest.raises(ValueError, match=expected_error):
        _make_neox_args(attn_softmax_clipping=True, **updates)


@pytest.mark.cpu
def test_attention_softmax_clipping_bounds_are_ignored_when_disabled():
    neox_args = _make_neox_args(
        attn_softmax_clipping=False,
        attn_softmax_clipping_zeta=0.5,
        attn_softmax_clipping_gamma=0.5,
    )

    assert neox_args.attn_softmax_clipping is False


@pytest.mark.cpu
def test_attention_softmax_clipping_rejects_unsupported_backends(monkeypatch):
    with pytest.raises(ValueError, match="not supported with `te_mha`"):
        _make_neox_args(attn_softmax_clipping=True, te_mha=True)

    with pytest.raises(ValueError, match="not supported by attention types: local"):
        _make_neox_args(
            attn_softmax_clipping=True,
            attention_config=[[["local"], "all"]],
        )

    monkeypatch.setattr(neox_arguments, "version", lambda _: "2.5.6")
    with pytest.raises(ValueError, match="not supported by attention types: flash"):
        _make_neox_args(
            attn_softmax_clipping=True,
            attention_config=[[["flash"], "all"]],
        )
