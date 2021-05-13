import os
import time
import shutil
import itertools
from pathlib import Path

import pytest
import random

import torch
import torch.distributed as dist
from torch.multiprocessing import Process

import deepspeed

TEST_CHECKPOINT_DIR = "test_checkpoint"
TEST_LOG_DIR = "test_logs"
TEST_TENSORBOARD_DIR = "test_tensorboard"

# Worker timeout *after* the first worker has completed.
DEEPSPEED_UNIT_WORKER_TIMEOUT = 120


def get_root_directory():
    return Path(__file__).parents[1]


def get_config_directory():
    return get_root_directory() / "configs"


def get_configs_with_path(configs):
    return [str(get_config_directory() / cfg) for cfg in configs]


def get_test_configs_with_path(configs):
    test_config_dir = Path(__file__).parent / "test_configs"
    return [str((test_config_dir / cfg).absolute()) for cfg in configs]


def clear_test_dirs():
    log_dir = os.path.join(get_root_directory(), TEST_LOG_DIR)
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)

    checkpoint_dir = os.path.join(get_root_directory(), TEST_CHECKPOINT_DIR)
    if os.path.isdir(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

    tensorboard_dir = os.path.join(get_root_directory(), TEST_TENSORBOARD_DIR)
    if os.path.isdir(tensorboard_dir):
        shutil.rmtree(tensorboard_dir)


def distributed_test(world_size=2, backend='nccl'):
    """A decorator for executing a function (e.g., a unit test) in a distributed manner.
    This decorator manages the spawning and joining of processes, initialization of
    torch.distributed, and catching of errors.

    This function is copied from: https://github.com/EleutherAI/DeeperSpeed/blob/24026e5bb37c528a222b8635c46256b1e1825d2e/tests/unit/common.py#L16

    Usage example:
        @distributed_test(worker_size=[2,3])
        def my_test():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            assert(rank < world_size)

    Arguments:
        world_size (int or list): number of ranks to spawn. Can be a list to spawn
        multiple tests.
    """

    def dist_wrap(run_func):
        """Second-level decorator for dist_test. This actually wraps the function. """

        def dist_init(local_rank, num_procs, *func_args, **func_kwargs):
            """Initialize torch.distributed and execute the user function. """
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '29503'
            os.environ['LOCAL_RANK'] = str(local_rank)
            # NOTE: unit tests don't support multi-node so local_rank == global rank
            os.environ['RANK'] = str(local_rank)
            os.environ['WORLD_SIZE'] = str(num_procs)

            deepspeed.init_distributed(dist_backend=backend)

            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)

            run_func(*func_args, **func_kwargs)

        def dist_launcher(num_procs, *func_args, **func_kwargs):
            """Launch processes and gracefully handle failures. """

            # Spawn all workers on subprocesses.
            processes = []
            for local_rank in range(num_procs):
                p = Process(target=dist_init,
                            args=(local_rank,
                                  num_procs,
                                  *func_args),
                            kwargs=func_kwargs)
                p.start()
                processes.append(p)

            # Now loop and wait for a test to complete. The spin-wait here isn't a big
            # deal because the number of processes will be O(#GPUs) << O(#CPUs).
            any_done = False
            while not any_done:
                for p in processes:
                    if not p.is_alive():
                        any_done = True
                        break

            # Wait for all other processes to complete
            for p in processes:
                p.join(DEEPSPEED_UNIT_WORKER_TIMEOUT)

            failed = [(rank, p) for rank, p in enumerate(processes) if p.exitcode != 0]
            for rank, p in failed:
                # If it still hasn't terminated, kill it because it hung.
                if p.exitcode is None:
                    p.terminate()
                    pytest.fail(f'Worker {rank} hung.', pytrace=False)
                if p.exitcode < 0:
                    pytest.fail(f'Worker {rank} killed by signal {-p.exitcode}',
                                pytrace=False)
                if p.exitcode > 0:
                    pytest.fail(f'Worker {rank} exited with code {p.exitcode}',
                                pytrace=False)

        def run_func_decorator(*func_args, **func_kwargs):
            """Entry point for @distributed_test(). """

            if isinstance(world_size, int):
                dist_launcher(world_size, *func_args, **func_kwargs)
            elif isinstance(world_size, list):
                for procs in world_size:
                    dist_launcher(procs, *func_args, **func_kwargs)
                    time.sleep(0.5)
            else:
                raise TypeError(f'world_size must be an integer or a list of integers.')

        return run_func_decorator

    return dist_wrap


def model_setup(yaml_list=None, param_dict=None):
    from megatron.neox_arguments import NeoXArgs
    from megatron.mpu import destroy_model_parallel
    from megatron import initialize_megatron
    from megatron.training import setup_model_and_optimizer

    destroy_model_parallel()  # mpu model parallel contains remaining global vars
    if torch.distributed.get_world_size() == 1 or torch.distributed.get_rank() == 0:
        clear_test_dirs()

    overwrite_values = {
        "user_script": str(get_root_directory() / "pretrain_gpt2.py"),
        "save": TEST_CHECKPOINT_DIR,
        "load": TEST_CHECKPOINT_DIR,
        "log_dir": TEST_LOG_DIR,
        "tensorboard_dir": TEST_TENSORBOARD_DIR,
    }

    # should not both be none
    assert yaml_list is not None or param_dict is not None

    # initially load config from files as would be the case in deepy.py
    if yaml_list is not None:
        args_loaded = NeoXArgs.from_ymls(yaml_list, overwrite_values=overwrite_values)
    else:
        p_dict = param_dict.copy()
        p_dict.update(overwrite_values)
        args_loaded = NeoXArgs.from_dict(p_dict)

    args_loaded.build_tokenizer()

    initialize_megatron(neox_args=args_loaded)
    model, optimizer, lr_scheduler = setup_model_and_optimizer(neox_args=args_loaded, inference=False,
                                                               get_key_value=True)
    return model, optimizer, lr_scheduler, args_loaded


def bounded_product(sequence, n=None, seed=None):
    """
    Returns a shuffled, bounded cartesian product of the input sequence.
    Designed to cover as wide a range of permutations as possible with a limited number of iterations.
    Will manifest the whole list in memory, so not suitable for super large sequences.

    :param sequence: iterable
    :param n: length of returned list
    :param seed: random seed for reproducibility
    :return: list
    """
    p = list(itertools.product(*sequence))
    if seed is not None:
        random.seed(seed)
    random.shuffle(p)
    return p if n is None else p[:n]


binary = [True, False]
