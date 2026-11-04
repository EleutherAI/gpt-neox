"""Exercise different routing labels on two real CPU DDP ranks."""
from pathlib import Path
import sys

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, str(Path(__file__).parent))
from test_routing import setup


def worker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", init_method=f"file://{rendezvous}",
                                         rank=rank, world_size=2)
    try:
        model, route, optimizer = setup()
        ddp = DistributedDataParallel(model)
        # A serial reference accumulates the two ranks' losses before one update.
        reference, ref_route, ref_optimizer = setup()
        x = torch.tensor([[1, 2, 3, 4]])
        for iteration, labels in enumerate([(True, True), (False, True), (False, False)]):
            optimizer.zero_grad(set_to_none=True)
            ref_optimizer.zero_grad(set_to_none=True)
            route.select(labels[rank], 0, 1234, iteration, 0)
            ddp(x).square().mean().backward()
            optimizer.step()
            for microstep, flagged in enumerate(labels):
                ref_route.select(flagged, 0, 1234, iteration, microstep)
                (reference(x).square().mean() / 2).backward()
            # Avoid the optimizer's all-reduce for the serial reference: same
            # activity on each rank means MAX leaves it unchanged.
            ref_optimizer.step()
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    finally:
        torch.distributed.destroy_process_group()


def test_mixed_rank_routing_matches_serial_reference(tmp_path):
    mp.spawn(worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


def nested_worker(rank, rendezvous):
    from test_nested import setup as nested_setup
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        model, route, optimizer = nested_setup()
        ddp = DistributedDataParallel(model)
        reference, ref_route, ref_optimizer = nested_setup()
        x = torch.tensor([[1, 2, 3, 4]])
        for iteration, labels in enumerate([(2, 2), (1, 2), (0, 2), (0, 1), (0, 0)]):
            optimizer.zero_grad(set_to_none=True)
            ref_optimizer.zero_grad(set_to_none=True)
            route.select(labels[rank], 0, 1234, iteration, 0)
            ddp(x).square().mean().backward()
            optimizer.step()
            for microstep, label in enumerate(labels):
                ref_route.select(label, 0, 1234, iteration, microstep)
                (reference(x).square().mean() / 2).backward()
            ref_optimizer.step()
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    finally:
        torch.distributed.destroy_process_group()


def test_nested_mixed_rank_routing_matches_serial_reference(tmp_path):
    mp.spawn(nested_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
