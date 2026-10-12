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
import math

import pytest
import torch

import megatron.model.gpt2_model as gpt2_model
from megatron.model.gpt2_model import GPT2ModelPipe
from megatron.neox_arguments import NeoXArgs
from tests.common import BASE_CONFIG


class CapturedRelativePositionBias:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def build_model_specs(neox_args):
    model = GPT2ModelPipe.__new__(GPT2ModelPipe)
    torch.nn.Module.__init__(model)
    model.neox_args = neox_args
    model.hidden_size = neox_args.hidden_size
    model.num_tokentypes = 0
    model.init_method = None
    model.output_layer_init_method = None
    model.parallel_output = True
    model.use_cache = False
    model.init_specs()
    return model.specs


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("head_dim", "expected_head_dim"),
    [(None, 4), (6, 6)],
)
def test_rpe_scale_uses_effective_head_dim(monkeypatch, head_dim, expected_head_dim):
    config = deepcopy(BASE_CONFIG)
    config.update(
        {
            "hidden_size": 8,
            "num_attention_heads": 2,
            "head_dim": head_dim,
            "pos_emb": "rpe",
            "num_layers": 1,
            "precision": "fp32",
            "global_num_gpus": 1,
        }
    )
    neox_args = NeoXArgs.from_dict(config)
    monkeypatch.setattr(
        gpt2_model,
        "ParallelRelativePositionBias",
        CapturedRelativePositionBias,
    )

    specs = build_model_specs(neox_args)
    transformer_spec = next(
        spec
        for spec in specs
        if getattr(spec, "typename", None) is gpt2_model.ParallelTransformerLayerPipe
    )
    rpe = transformer_spec.module_kwargs["rpe"]

    assert rpe.scale == math.sqrt(expected_head_dim)
    assert rpe.heads == neox_args.num_attention_heads
