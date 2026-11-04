"""Optional tests of actual DeepSpeed wrappers, using only local CPU ranks.

Run with GRAM_TEST_DEEPSPEED=1 DS_ACCELERATOR=cpu and a DeeperSpeed installation.
"""
import copy
from datetime import timedelta
import importlib.util
import os
from pathlib import Path
import sys

import pytest
import torch
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent))
from test_routing import Network, args, gram


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_equal(x, y)
    else:
        assert a == b

pytestmark = pytest.mark.skipif(
    os.environ.get("GRAM_TEST_DEEPSPEED") != "1" or importlib.util.find_spec("deepspeed") is None,
    reason="Set GRAM_TEST_DEEPSPEED=1 with DeeperSpeed installed for CPU wrapper tests",
)


def worker(rank, rendezvous, zero_stage):
    os.environ.update(DS_ACCELERATOR="cpu", LOCAL_RANK=str(rank), RANK=str(rank), WORLD_SIZE="2")
    torch.set_num_threads(1)
    import deepspeed
    # Use standard Gloo rather than compiling the optional shared-memory backend.
    deepspeed.ops.__compatible_ops__["deepspeed_shm_comm"] = False
    deepspeed.init_distributed("gloo", init_method=f"file://{rendezvous}", rank=rank,
                              world_size=2, timeout=timedelta(seconds=60), auto_mpi_discovery=False)
    try:
        torch.manual_seed(7)
        model = Network()
        route = gram.install(model, args())
        optimizer = gram.RoutedAdamW(model.parameters(), route, managed_step=True,
                                     lr=0.01, weight_decay=0.1)
        config = {"train_micro_batch_size_per_gpu": 1, "gradient_accumulation_steps": 2,
                    "bf16": {"enabled": True}, "data_types": {"grad_accum_dtype": "fp32"},
                    "zero_optimization": {"stage": zero_stage, "overlap_comm": False},
                    "zero_allow_untested_optimizer": True, "steps_per_print": 1000}
        engine, _, _, _ = deepspeed.initialize(
            model=model, optimizer=optimizer, dist_init_required=False, config=config,
        )
        original = {n: p.detach().clone() for n, p in model.named_parameters()}
        frozen_aux = None
        frozen_aux_state = None
        for iteration, labels in enumerate([(True, True), (False, False), (False, True)]):
            for micro in range(2):
                route.select(labels[rank], 0, 1234, iteration, micro)
                x = torch.tensor([[1, 2, 3, 4]])
                loss = engine(x).float().square().mean()
                engine.backward(loss)
                boundary = engine.is_gradient_accumulation_boundary()
                if boundary:
                    route.prepare_step(engine.device)
                engine.step()
                if boundary:
                    route.finish_step()
            if iteration == 0:
                for name, p in model.named_parameters():
                    if not p.gram_role:
                        torch.testing.assert_close(p, original[name], rtol=0, atol=0)
                frozen_aux = {n: p.detach().clone() for n, p in model.named_parameters() if p.gram_role}
                frozen_aux_state = [copy.deepcopy(optimizer.state[g["params"][0]])
                                    for g in optimizer.param_groups if g["gram_role"]]
                assert all(optimizer.state[g["params"][0]]["step"].item() == g["gram_role"]
                           for g in optimizer.param_groups)
                engine.save_checkpoint(str(Path(rendezvous).parent / "checkpoint"), tag="step1")
            if iteration == 1:
                for name, p in model.named_parameters():
                    if p.gram_role:
                        torch.testing.assert_close(p, frozen_aux[name], rtol=0, atol=0)
                for old, group in zip(frozen_aux_state,
                                      [g for g in optimizer.param_groups if g["gram_role"]]):
                    for key, value in old.items():
                        torch.testing.assert_close(optimizer.state[group["params"][0]][key],
                                                   value, rtol=0, atol=0)
            for p in model.parameters():
                peer = p.detach().clone()
                torch.distributed.broadcast(peer, src=0)
                torch.testing.assert_close(p, peer, rtol=0, atol=0)
        assert all(optimizer.state[g["params"][0]]["step"].item() == 2 for g in optimizer.param_groups)
        expected_model = copy.deepcopy(model.state_dict())
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        # Rebuild all components and restore the actual DeepSpeed checkpoint,
        # including every rank's ZeRO partition and previously inactive state.
        other = Network()
        restored_route = gram.install(other, args())
        restored_opt = gram.RoutedAdamW(other.parameters(), restored_route, managed_step=True,
                                        lr=0.01, weight_decay=0.1)
        restored, _, _, _ = deepspeed.initialize(
            model=other, optimizer=restored_opt, dist_init_required=False, config=config)
        loaded, _ = restored.load_checkpoint(str(Path(rendezvous).parent / "checkpoint"), tag="step1")
        assert loaded is not None
        for iteration, labels in [(1, (False, False)), (2, (False, True))]:
            for micro in range(2):
                restored_route.select(labels[rank], 0, 1234, iteration, micro)
                restored.backward(restored(torch.tensor([[1, 2, 3, 4]])).float().square().mean())
                boundary = restored.is_gradient_accumulation_boundary()
                if boundary:
                    restored_route.prepare_step(restored.device)
                restored.step()
                if boundary:
                    restored_route.finish_step()
        assert_equal(expected_model, other.state_dict())
        assert_equal(expected_optimizer, restored_opt.state_dict())
        assert engine.global_steps == restored.global_steps
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("zero_stage", [0, 1])
def test_actual_bf16_wrapper_and_role_freezing(tmp_path, zero_stage):
    mp.spawn(worker, args=(str(tmp_path / "rendezvous"), zero_stage), nprocs=2, join=True)
