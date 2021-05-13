"""
instantiate models, save checkpoints, load checkpoints, compare loaded parameters to saved parameters and compare forward pass outputs

This tests contain a relatively large number of functions. They are not split into separate tests because a lot of boilerplate (e.g. instantiate model) needs
to run in order to perform follow up tests. Joining in one test reduces runtime at the expense of decreased transparency of test results in case of failures.
"""
import pytest

from copy import deepcopy
from ..common import distributed_test, clear_test_dirs, model_setup, get_test_configs_with_path, bounded_product, binary

import torch
from yaml import load

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

with open(get_test_configs_with_path("test_train_base.yml")[0], 'r') as f:
    BASE_CONFIG = load(f, Loader=Loader)

PARAMS_TO_TEST = {
    "norm,pos_emb": [["layernorm", "learned"], ["rmsnorm", "rotary"], ["scalenorm", "sinusoidal"],
                     ["layernorm", "rpe"], ["rmsnorm", "none"]],
    "pipe_parallel_size,model_parallel_size": [[0, 1], [1, 1], [2, 2], [0, 2]],
    "no_weight_tying": binary,
    "attention_config": [[[["global"], "all"]], [[["local"], "all"]], [[["sparse_variable"], "all"]],
                         [[["sparse_fixed"], "all"]]],
    "scaled_upper_triang_masked_softmax_fusion": binary,
    "bias_gelu_fusion": binary,
    "checkpoint_activations": binary,
}


def parametrize(params_to_test: dict, max_tests: int = 50, seed: int = None):
    """
    Generates a random sample of max_tests length of all possible combinations of values in
    `params_to_test`.

    In `params_to_test` you can either specify one value, and all possible settings of that value,
    or two values separated by a comma, and all possible combinations of those two values in tandem.
        i.e "hidden_size,num_heads": [[768,12], [1024,32], [2048, 64]]
    so the first item in each list is a value of `hidden_size` and the second a value of `num_heads`
    this is useful for reducing the size of possible tests for values we know are unlikely to interact beforehand,
    since the cartesian product can grow very large.

    :param params_to_test: dict of neox params
    :param max_tests: maximum number of tests to run
    :param seed: random seed
    :return: a list of neox param dicts to pass to a parametrized unit test
    """
    keys, values = zip(*params_to_test.items())
    for p in bounded_product(values, n=max_tests, seed=seed):
        experiment = dict(zip(keys, p))
        for k, v in experiment.items():
            if "," in k:
                keys_split = [i.strip() for i in k.split(',')]
                values_separated = experiment.pop(k)
                assert len(values_separated) == len(keys_split)
                new_dict = dict(zip(keys_split, values_separated))
                experiment.update(new_dict)
        base = deepcopy(BASE_CONFIG)
        base.update(experiment)
        yield base


@pytest.mark.parametrize("param_dict", list(parametrize(PARAMS_TO_TEST, max_tests=50, seed=None)))
@distributed_test(world_size=2)
def test_train(param_dict):
    run_train_test(param_dict=param_dict)


def run_train_test(yaml_list=None, param_dict=None):
    from megatron.training import train_step
    from megatron.utils import Timers

    max_steps = 256

    model, optimizer, lr_scheduler, args_loaded = model_setup(yaml_list, param_dict)

    model.train()

    timers = Timers(use_wandb=False, tensorboard_writer=None)

    # generate some random data on which we can overfit
    # context size of data is model seq_len + 1 in order to compute loss
    data_list = list()
    context_tokens_tensor = torch.randint(0, args_loaded.padded_vocab_size, (4, args_loaded.seq_length + 1)).to(
        torch.int64)
    for i in range(max_steps):
        data_list.append({"text": context_tokens_tensor.clone()})
    data_iterator = iter(data_list)

    # run train_step until the loss decreases
    losses = list()
    for i in range(max_steps):
        loss_dict, skipped_iter = train_step(
            neox_args=args_loaded,
            timers=timers,
            data_iterator=data_iterator,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler
        )
        losses.append(loss_dict["lm_loss"])
        if len(losses) >= 2:
            if torch.isnan(losses[-1]): continue
            if torch.isnan(losses[-2]): continue
            if losses[-1] < losses[-2]:
                return  # all good

    # loss should have decreased by now (otherwise increasing the max_steps parameter could have the testcase pass)
    assert losses[-1] < losses[-2], "run_train_test() loss going down within " + str(max_steps) + " steps"

    if torch.distributed.get_world_size() == 1 or torch.distributed.get_rank() == 0:
        clear_test_dirs()
