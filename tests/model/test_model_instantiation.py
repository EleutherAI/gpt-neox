"""
instantiate models with different configurations as a first possible point of failure
"""

import pytest

import torch

from ..common import distributed_test, model_setup, clear_test_dirs, parametrize, binary

PARAMS_TO_TEST = {
    "pipe_parallel_size,model_parallel_size": [[0, 1], [1, 2], [0, 2]],
    "no_weight_tying": binary,
    "attention_config": [[[["global"], "all"]], [[["local"], "all"]], [[["sparse_variable"], "all"]],
                         [[["sparse_fixed"], "all"]]],
    "scaled_upper_triang_masked_softmax_fusion,bias_gelu_fusion": [[True, False], [False, True]],
}


@pytest.mark.parametrize("param_dict", list(parametrize(PARAMS_TO_TEST, max_tests=50, seed=None)))
def test_train(param_dict):
    @distributed_test(world_size=2)
    def wrapper():
        run_test_model_instantiation(param_dict=param_dict)
    wrapper()


def run_test_model_instantiation(yaml_list=None, param_dict=None):
    from deepspeed.runtime.pipe.engine import PipelineEngine, DeepSpeedEngine
    model, optimizer, lr_scheduler, args_loaded = model_setup(yaml_list, param_dict)
    print(type(model), flush=True)
    if args_loaded.pipe_parallel_size < 2:
        assert isinstance(model, DeepSpeedEngine), "test model instantiation " + str(yaml_list)
    else:
        assert isinstance(model, PipelineEngine), "test model instantiation " + str(yaml_list)

    if torch.distributed.get_world_size() == 1 or torch.distributed.get_rank() == 0:
        clear_test_dirs()
