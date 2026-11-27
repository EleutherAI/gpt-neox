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

import math
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron import mpu
from megatron.model import gpt2_model
from megatron.training import evaluate, forward_step, train_step

pytestmark = pytest.mark.cpu


def _check_sharded_lm_loss(
    rank, world_size, device, dtype, fused, coefficient, fp16_ce=False
):
    # Targets and token maxima lie on different shards. Large offsets exercise
    # logsumexp stability, and the largest offset belongs to a masked token.
    logits = torch.tensor(
        [
            [1, 3, 2, -1, 0, 1],
            [-2, 0, 1, 2, 3, 0],
            [1, 0, 2, 3, 0, 1],
            [0, -50, -50, -50, -50, -50],
        ],
        device=device,
        dtype=dtype,
    )
    logits = (logits + logits.new_tensor([[300], [400], [-100], [1e-7]])).unsqueeze(0)
    labels = torch.tensor([[4, 1, 0, 0]], device=device)
    mask = torch.tensor([[1.0, 0.0, 0.5, 1.0]], device=device)
    if fp16_ce:
        mask = mask.half()

    # Keep the reference logsumexp gradient accurate at large offsets.
    reference = logits.detach().double().requires_grad_()
    token_losses = F.cross_entropy(reference.transpose(1, 2), labels, reduction="none")
    penalty = reference.logsumexp(dim=-1).square()
    expected = ((token_losses + coefficient * penalty) * mask).sum() / mask.sum()
    expected.backward()

    shard = logits.chunk(world_size, dim=-1)[rank].contiguous().requires_grad_()
    args = SimpleNamespace(
        train_impl="normal",
        memory_profiling=False,
        curriculum_learning=False,
        is_pipe_parallel=False,
        fp16_lm_cross_entropy=fp16_ce,
        cross_entropy_fusion=fused,
        z_loss=coefficient,
    )
    saved_shards = []

    def record_saved_tensor(tensor):
        if tensor.shape == shard.shape:
            saved_shards.append(tensor.dtype)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(
        record_saved_tensor, lambda tensor: tensor
    ), patch.object(dist, "all_reduce", wraps=dist.all_reduce) as all_reduce, patch(
        "megatron.training.get_batch", return_value=(None, labels, mask, None, None)
    ):
        loss, metrics = forward_step(
            None, Mock(return_value=shard), args, timers=None, is_train=True
        )
        loss.backward()

    work_dtype = torch.float16 if fp16_ce and coefficient == 0.0 else torch.float32
    assert all_reduce.call_count == (2 if fused else 3)
    assert saved_shards == [work_dtype]
    torch.testing.assert_close(loss, expected.to(work_dtype))
    if coefficient:
        assert set(metrics) == {"lm_loss", "z_loss"}
        assert not any(value.requires_grad for value in metrics.values())
        torch.testing.assert_close(
            metrics["lm_loss"], ((token_losses * mask).sum() / mask.sum()).float()
        )
        torch.testing.assert_close(
            metrics["z_loss"], ((penalty * mask).sum() / mask.sum()).float()
        )
        eval_logits = logits.chunk(world_size, dim=-1)[rank].contiguous().clone()
        with torch.no_grad(), patch(
            "megatron.training.get_batch", return_value=(None, labels, mask, None, None)
        ):
            eval_loss, returned_logits, eval_metrics = forward_step(
                None,
                Mock(return_value=eval_logits),
                args,
                timers=None,
                return_logits=True,
            )
        assert eval_metrics == {}
        torch.testing.assert_close(
            returned_logits, reference.chunk(world_size, dim=-1)[rank].to(dtype)
        )
        torch.testing.assert_close(eval_loss, metrics["lm_loss"].to(eval_loss.dtype))
    else:
        assert metrics == {}
    expected_grad = reference.grad.chunk(world_size, dim=-1)[rank].to(dtype)
    torch.testing.assert_close(shard.grad, expected_grad)
    if rank == 0 and dtype == torch.float32:
        # The last target has probability one in FP32 and a tiny Z-loss gradient.
        torch.testing.assert_close(
            shard.grad[0, -1, 0], expected_grad[0, -1, 0], rtol=1e-5, atol=0
        )


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
        for coefficient in (0.0, 0.025):
            _check_sharded_lm_loss(
                rank,
                world_size,
                device,
                torch.float16,
                False,
                coefficient,
                fp16_ce=True,
            )
        for coefficient in (0.0, 0.025, 0.1):
            _check_training_and_eval_metrics(device, coefficient)
        # Validation with native FP16 CE must also handle a long, fully unmasked batch.
        labels = torch.zeros((1, 65536), device=device, dtype=torch.long)
        mask = torch.ones_like(labels, dtype=torch.float16)
        logits = torch.zeros((*labels.shape, 2), device=device, dtype=torch.float16)
        loss = gpt2_model.cross_entropy(logits, (labels, mask), _fp16=True)
        torch.testing.assert_close(loss, loss.new_tensor(math.log(4)))
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


def _check_training_and_eval_metrics(device, coefficient):
    args = SimpleNamespace(
        train_impl="normal",
        is_pipe_parallel=False,
        memory_profiling=False,
        curriculum_learning=False,
        fp16_lm_cross_entropy=False,
        cross_entropy_fusion=True,
        deepspeed=True,
        deepspeed_activation_checkpointing=False,
        precision="fp32",
        profile=False,
        gradient_accumulation_steps=2,
        eval_iters=1,
        eval_tasks=[],
        char_level_ppl=False,
        z_loss=coefficient,
    )
    labels = torch.tensor([[1]], device=device)
    batch = (None, labels, torch.ones_like(labels).float(), None, None)
    # Both microbatches predict a uniform distribution over four classes. Only Z-loss sees the shift.
    model = Mock(
        side_effect=[
            torch.full((1, 1, 2), offset, device=device, requires_grad=True)
            for offset in (0.0, 5.0, 0.0, 5.0)
        ]
    )
    model.backward.side_effect = lambda loss: loss.backward()
    with patch("megatron.training.get_batch", return_value=batch):
        metrics, _ = train_step(args, Mock(), None, model, None, None)
        results = evaluate(args, forward_step, None, model)
    ce = math.log(4)
    expected = {"lm_loss": ce}
    if coefficient:
        penalty = (ce**2 + (ce + 5) ** 2) / 2
        expected.update(z_loss=penalty, lm_loss_with_z_loss=ce + coefficient * penalty)
    assert {k: v.item() for k, v in metrics.items()} == pytest.approx(expected)
    assert results == pytest.approx({"lm_loss": ce, "lm_loss_ppl": 4.0})


@pytest.mark.parametrize("coefficient", [0.0, 0.125])
def test_pipeline_loss_uses_configured_z_loss_only_in_training(
    monkeypatch, coefficient
):
    from megatron.neox_arguments import NeoXArgs
    from tests.common import BASE_CONFIG

    args = NeoXArgs.from_dict(
        dict(
            BASE_CONFIG,
            global_num_gpus=1,
            z_loss=coefficient,
            cross_entropy_fusion=True,
        )
    )

    def pipeline_init(module, **kwargs):
        torch.nn.Module.__init__(module)
        module.loss_fn = kwargs["loss_fn"]

    cross_entropy = Mock()
    monkeypatch.setattr(gpt2_model.PipelineModule, "__init__", pipeline_init)
    monkeypatch.setattr(gpt2_model, "cross_entropy", cross_entropy)
    model = gpt2_model.GPT2ModelPipe(args)
    logits, labels = object(), object()
    for training in (True, False, True):
        model.train(training)
        model.loss_fn(logits, labels)
        additional = model.get_additional_losses()
        if training and coefficient:
            assert set(additional) == {"lm_loss", "z_loss"}
        else:
            assert additional is None
        cross_entropy.assert_called_with(
            logits,
            labels,
            _fp16=False,
            _fused=True,
            z_loss=coefficient if training else 0.0,
            metrics=additional,
            preserve_logits=not training,
        )


@pytest.mark.parametrize("coefficient", [0.0, 0.125])
@pytest.mark.parametrize("is_train", [False, True])
def test_reward_model_z_loss_uses_selected_scores(coefficient, is_train):
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
        loss, metrics = forward_step(
            None, Mock(return_value=scores), args, timers=None, is_train=is_train
        )

    reference = scores.detach().clone().requires_grad_()
    chosen, rejected = reference[0, 1, 0], reference[1, 0, 0]
    expected = F.softplus(rejected - chosen)
    if is_train and coefficient:
        penalty = chosen.square() + rejected.square()
        torch.testing.assert_close(metrics["lm_loss"], expected)
        torch.testing.assert_close(metrics["z_loss"], penalty)
        expected = expected + coefficient * penalty
    else:
        assert "z_loss" not in metrics
    expected.backward()
    loss.backward()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(scores.grad, reference.grad)
