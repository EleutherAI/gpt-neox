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

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron import mpu
from megatron.model import gpt2_model
from megatron.training import forward_step

pytestmark = pytest.mark.cpu


def _check_sharded_lm_loss(rank, world_size, device, dtype, fused, coefficient):
    # Targets and token maxima lie on different shards. Large offsets exercise
    # logsumexp stability, and the largest offset belongs to a masked token.
    logits = torch.tensor(
        [[1, 3, 2, -1, 0, 1], [-2, 0, 1, 2, 3, 0], [1, 0, 2, 3, 0, 1]],
        device=device,
        dtype=dtype,
    )
    logits = (logits + logits.new_tensor([[100], [300], [-100]])).unsqueeze(0)
    labels = torch.tensor([[4, 1, 0]], device=device)
    mask = torch.tensor([[1.0, 0.0, 1.0]], device=device)

    reference = logits.detach().float().clone().requires_grad_()
    token_losses = F.cross_entropy(reference.transpose(1, 2), labels, reduction="none")
    penalty = reference.logsumexp(dim=-1).square()
    expected = (token_losses + coefficient * penalty)[mask.bool()].mean()
    expected.backward()

    shard = logits.chunk(world_size, dim=-1)[rank].contiguous().requires_grad_()
    args = SimpleNamespace(
        train_impl="normal",
        memory_profiling=False,
        curriculum_learning=False,
        is_pipe_parallel=False,
        fp16_lm_cross_entropy=False,
        cross_entropy_fusion=fused,
        z_loss=coefficient,
    )
    with patch(
        "megatron.training.get_batch", return_value=(None, labels, mask, None, None)
    ):
        loss, _ = forward_step(
            None, Mock(return_value=shard), args, timers=None, is_train=True
        )
    loss.backward()

    torch.testing.assert_close(loss, expected)
    expected_grad = reference.grad.chunk(world_size, dim=-1)[rank].to(dtype)
    torch.testing.assert_close(shard.grad, expected_grad)


def _run_tensor_parallel_lm_loss(rank, store_path, world_size, use_cuda):
    device = torch.device("cuda", rank) if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl" if use_cuda else "gloo",
        init_method=store_path.as_uri(),
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        mpu.initialize_model_parallel(world_size)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for fused in (False, True):
                for coefficient in (0.0, 0.025):
                    _check_sharded_lm_loss(
                        rank, world_size, device, dtype, fused, coefficient
                    )
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


def test_tensor_parallel_lm_loss_matches_dense_loss_and_gradients(tmp_path):
    world_size = 2
    torch.multiprocessing.spawn(
        _run_tensor_parallel_lm_loss,
        args=(
            tmp_path / "tp-store",
            world_size,
            torch.cuda.device_count() >= world_size,
        ),
        nprocs=world_size,
    )


def test_pipeline_loss_uses_configured_z_loss(monkeypatch):
    from megatron.neox_arguments import NeoXArgs
    from tests.common import BASE_CONFIG

    args = NeoXArgs.from_dict(
        dict(BASE_CONFIG, global_num_gpus=1, z_loss=0.125, cross_entropy_fusion=True)
    )
    pipeline_init = Mock(return_value=None)
    cross_entropy = Mock()
    monkeypatch.setattr(gpt2_model.PipelineModule, "__init__", pipeline_init)
    monkeypatch.setattr(gpt2_model, "cross_entropy", cross_entropy)
    gpt2_model.GPT2ModelPipe(args)

    loss_fn = pipeline_init.call_args.kwargs["loss_fn"]
    logits, labels = object(), object()
    loss_fn(logits, labels)
    cross_entropy.assert_called_once_with(
        logits, labels, _fp16=False, _fused=True, z_loss=0.125
    )


@pytest.mark.parametrize("coefficient", [0.0, 0.125])
def test_reward_model_z_loss_uses_selected_scores(coefficient):
    args = SimpleNamespace(
        train_impl="rm",
        is_pipe_parallel=False,
        memory_profiling=False,
        z_loss=coefficient,
    )
    scores = torch.tensor([[[8.0], [2.0]], [[-1.0], [6.0]]], requires_grad=True)
    mask = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    batch = (None, None, mask, None, None, None)
    with patch("megatron.training.get_batch", return_value=batch):
        loss, _ = forward_step(None, Mock(return_value=scores), args, timers=None)

    reference = scores.detach().clone().requires_grad_()
    chosen, rejected = reference[0, 1, 0], reference[1, 0, 0]
    expected = F.softplus(rejected - chosen) + coefficient * (
        chosen.square() + rejected.square()
    )
    expected.backward()
    loss.backward()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(scores.grad, reference.grad)
