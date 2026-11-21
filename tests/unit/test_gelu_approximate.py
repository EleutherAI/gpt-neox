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

from contextlib import contextmanager
from copy import deepcopy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron import mpu
from megatron.model.activations import bias_gelu_impl, get_activation
from megatron.model.transformer import ParallelMLP
from megatron.neox_arguments import NeoXArgs
from tests.common import BASE_CONFIG

pytestmark = pytest.mark.cpu


def _make_neox_args(**updates):
    config = deepcopy(BASE_CONFIG)
    config["global_num_gpus"] = 1
    config["use_cpu_initialization"] = True
    config["precision"] = "fp32"
    config.update(updates)
    return NeoXArgs.from_dict(config)


@contextmanager
def _single_rank_model_parallel(tmp_path):
    store = dist.FileStore(str(tmp_path / "store"), 1)
    dist.init_process_group(
        backend="gloo",
        store=store,
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=60),
    )
    try:
        mpu.initialize_model_parallel(1)
        yield
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


# Values where the exact and tanh-approximate GELU differ by more than float32
# rounding, so the tests can tell them apart.
X = torch.linspace(-4.0, 4.0, steps=33)
EXACT = F.gelu(X)
TANH = F.gelu(X, approximate="tanh")


def test_gelu_approximate_defaults_to_none():
    assert _make_neox_args().gelu_approximate == "none"


def test_gelu_approximate_rejects_unknown_values():
    with pytest.raises(ValueError, match="types cannot be validated"):
        _make_neox_args(gelu_approximate="sigmoid")


def test_tanh_gelu_is_rejected_with_onnx_safe():
    with pytest.raises(ValueError, match="onnx_safe"):
        _make_neox_args(activation="gelu", gelu_approximate="tanh", onnx_safe=True)


def test_inputs_separate_exact_and_tanh_gelu():
    assert not torch.allclose(EXACT, TANH, rtol=0, atol=1e-5)


@pytest.mark.parametrize("activation", ["gelu", "geglu"])
@pytest.mark.parametrize(
    ("gelu_approximate", "expected"), [("none", EXACT), ("tanh", TANH)]
)
def test_get_activation_uses_gelu_approximate(activation, gelu_approximate, expected):
    neox_args = _make_neox_args(
        activation=activation, gelu_approximate=gelu_approximate
    )

    activation_func, is_gated = get_activation(neox_args)

    assert is_gated is (activation == "geglu")
    torch.testing.assert_close(activation_func(X), expected, rtol=0, atol=0)


def test_bias_gelu_fusion_still_takes_precedence():
    neox_args = _make_neox_args(
        activation="gelu", gelu_approximate="none", bias_gelu_fusion=True
    )

    activation_func, _ = get_activation(neox_args)

    assert activation_func is bias_gelu_impl


def test_geglu_mlp_matches_gemma_mlp(tmp_path):
    # Hugging Face's GemmaMLP computes
    #   down_proj(gelu_pytorch_tanh(gate_proj(x)) * up_proj(x)).
    # NeoX fuses up_proj and gate_proj into linear1, in that order, and its
    # intermediate_size is 3x the HF one for gated activations.
    hidden_size, hf_intermediate_size = 8, 32
    neox_args = _make_neox_args(
        activation="geglu",
        gelu_approximate="tanh",
        intermediate_size=3 * hf_intermediate_size,
        use_bias_in_mlp=False,
        mlp_multiple_of=1,
    )

    with _single_rank_model_parallel(tmp_path):
        mlp = ParallelMLP(
            neox_args=neox_args,
            init_method=torch.nn.init.normal_,
            output_layer_init_method=torch.nn.init.normal_,
            multiple_of=neox_args.mlp_multiple_of,
        )
        x = torch.randn(3, 2, hidden_size)
        output, _ = mlp(x)

    up_proj, gate_proj = mlp.linear1.weight.chunk(2, dim=0)
    assert up_proj.shape == (hf_intermediate_size, hidden_size)
    expected = F.linear(
        F.gelu(F.linear(x, gate_proj), approximate="tanh") * F.linear(x, up_proj),
        mlp.linear2.weight,
    )
    torch.testing.assert_close(output, expected)
