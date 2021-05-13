"""
instantiate models with different configurations as a first possible point of failure
"""

import pytest

import torch

from ..common import distributed_test, get_test_configs_with_path, model_setup, clear_test_dirs

@distributed_test(world_size=1)
def test_model_instantiation_small_0():
    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_0.yml"])
    run_test_model_instantiation(yaml_list=yaml_list)

@distributed_test(world_size=1)
def test_model_instantiation_small_1():
    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_1.yml"])
    run_test_model_instantiation(yaml_list=yaml_list)

@distributed_test(world_size=2)
def test_model_instantiation_small_2():
    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_2.yml"])
    run_test_model_instantiation(yaml_list=yaml_list)

@distributed_test(world_size=1)
def test_model_instantiation_small_3():
    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_3.yml"])
    run_test_model_instantiation(yaml_list=yaml_list)

@distributed_test(world_size=2)
def test_model_instantiation_small_4():
    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_4.yml"])
    run_test_model_instantiation(yaml_list=yaml_list)

def run_test_model_instantiation(yaml_list=None, param_dict=None):
    from deepspeed.runtime.pipe.engine import PipelineEngine, DeepSpeedEngine
    model, optimizer, lr_scheduler, args_loaded = model_setup(yaml_list, param_dict)
    print(type(model), flush=True)
    if args_loaded.pipe_parallel_size < 2:
        assert isinstance(model, DeepSpeedEngine), "test model instantiation "+str(yaml_list)
    else:
        assert isinstance(model, PipelineEngine), "test model instantiation "+str(yaml_list)

    if torch.distributed.get_world_size() == 1 or torch.distributed.get_rank() == 0:
        clear_test_dirs()