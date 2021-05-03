"""
instantiate models with different configurations as a first possible point of failure
"""

import pytest

import torch

from ..common import TEST_CHECKPOINT_DIR, TEST_LOG_DIR, TEST_TENSORBOARD_DIR
from ..common import distributed_test, get_test_configs_with_path, get_root_directory, clear_test_dirs

@distributed_test(world_size=1)
def test_model_instantiation_small_0():
    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_0.yml"])
    run_test_model_instantiation(yaml_list)

@distributed_test(world_size=1)
def test_model_instantiation_small_1():
    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_1.yml"])
    run_test_model_instantiation(yaml_list)

@distributed_test(world_size=2)
def test_model_instantiation_small_2():
    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_2.yml"])
    run_test_model_instantiation(yaml_list)

@distributed_test(world_size=1)
def test_model_instantiation_small_3():
    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_3.yml"])
    run_test_model_instantiation(yaml_list)

@distributed_test(world_size=2)
def test_model_instantiation_small_4():
    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_4.yml"])
    run_test_model_instantiation(yaml_list)

def run_test_model_instantiation(yaml_list):
    from deepspeed.runtime.pipe.engine import PipelineEngine, DeepSpeedEngine

    from megatron.neox_arguments import NeoXArgs
    from megatron.mpu import destroy_model_parallel
    from megatron import initialize_megatron
    from megatron.training import setup_model_and_optimizer

    destroy_model_parallel() # mpu model parallel contains remaining global vars
    if torch.distributed.get_world_size() == 1 or torch.distributed.get_rank() == 0:
        clear_test_dirs()

    args_loaded = NeoXArgs.from_ymls(yaml_list, overwrite_values={
        "user_script": str(get_root_directory() / "pretrain_gpt2.py"),
        "use_cpu_initialization": True,
        "save": TEST_CHECKPOINT_DIR,
        "load": TEST_CHECKPOINT_DIR,
        "log_dir": TEST_LOG_DIR,
        "tensorboard_dir": TEST_TENSORBOARD_DIR,
    })
    args_loaded.build_tokenizer()

    initialize_megatron(neox_args=args_loaded)
    model, optimizer, lr_scheduler = setup_model_and_optimizer(neox_args=args_loaded, inference=False, get_key_value=True)
    
    print(type(model), flush=True)
    if args_loaded.pipe_parallel_size < 2:
        assert isinstance(model, DeepSpeedEngine), "test model instantiation "+str(yaml_list)
    else:
        assert isinstance(model, PipelineEngine), "test model instantiation "+str(yaml_list)

    if torch.distributed.get_world_size() == 1 or torch.distributed.get_rank() == 0:
        clear_test_dirs()