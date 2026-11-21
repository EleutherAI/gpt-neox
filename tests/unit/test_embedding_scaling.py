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

from megatron import mpu
from megatron.model.word_embeddings import Embedding
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


def _embed(tmp_path, neox_args, input_ids, dtype=torch.float32):
    with _single_rank_model_parallel(tmp_path):
        embedding = Embedding(
            neox_args=neox_args,
            hidden_size=neox_args.hidden_size,
            vocab_size=16,
            max_sequence_length=neox_args.max_position_embeddings,
            embedding_dropout_prob=0.0,
            init_method=torch.nn.init.normal_,
        ).to(dtype)
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
        output = embedding(input_ids, position_ids)
    return embedding.word_embeddings.weight, output


def test_scale_embeddings_defaults_to_false():
    assert _make_neox_args().scale_embeddings_by_sqrt_hidden_size is False


def test_embeddings_are_unscaled_by_default(tmp_path):
    neox_args = _make_neox_args()
    input_ids = torch.tensor([[0, 3, 7, 15]])

    weight, output = _embed(tmp_path, neox_args, input_ids)

    torch.testing.assert_close(output, weight[input_ids], rtol=0, atol=0)


def test_embeddings_are_scaled_by_sqrt_hidden_size(tmp_path):
    neox_args = _make_neox_args(scale_embeddings_by_sqrt_hidden_size=True)
    input_ids = torch.tensor([[0, 3, 7, 15]])

    weight, output = _embed(tmp_path, neox_args, input_ids)

    expected = weight[input_ids] * neox_args.hidden_size**0.5
    torch.testing.assert_close(output, expected)


def test_embedding_multiplier_is_rounded_to_the_embedding_dtype(tmp_path):
    # Gemma 7B: sqrt(3072) = 55.43 is 55.5 in bf16. Hugging Face multiplies by
    # the bf16 value, so NeoX should too.
    neox_args = _make_neox_args(
        scale_embeddings_by_sqrt_hidden_size=True,
        hidden_size=3072,
        num_attention_heads=16,
    )
    input_ids = torch.tensor([[0, 3, 7, 15]])

    weight, output = _embed(tmp_path, neox_args, input_ids, dtype=torch.bfloat16)

    assert weight.dtype == torch.bfloat16
    multiplier = torch.tensor(3072**0.5, dtype=torch.bfloat16)
    assert multiplier.item() == 55.5
    torch.testing.assert_close(output, weight[input_ids] * multiplier, rtol=0, atol=0)


def test_hf_export_rejects_scaled_embeddings():
    from tools.ckpts import convert_neox_to_hf

    with pytest.raises(
        NotImplementedError, match="scale_embeddings_by_sqrt_hidden_size"
    ):
        convert_neox_to_hf.convert(
            input_checkpoint_path=None,
            loaded_config={
                "norm": "rmsnorm",
                "scale_embeddings_by_sqrt_hidden_size": True,
            },
            output_checkpoint_path=None,
            architecture="llama",
        )
