"""
instantiate models, save checkpoints, load checkpoints, compare loaded parameters to saved parameters and compare forward pass outputs

This tests contain a relatively large number of functions. They are not split into separate tests because a lot of boilerplate (e.g. instantiate model) needs
to run in order to perform follow up tests. Joining in one test reduces runtime at the expense of decreased transparency of test results in case of failures.
"""
import pytest

from ..common import distributed_test, clear_test_dirs, model_setup, binary, parametrize

import torch

PARAMS_TO_TEST = {
    "norm,pos_emb,activation": [["layernorm", "learned", "gelu"], ["rmsnorm", "rotary", "gelu"],
                                ["scalenorm", "sinusoidal", "geglu"], ["layernorm", "rpe", "geglu"],
                                ["rmsnorm", "none", "geglu"]],
    "pipe_parallel_size,model_parallel_size": [[0, 1], [1, 2], [0, 2]],
    "no_weight_tying": binary,
    "attention_config": [[[["global"], "all"]], [[["local"], "all"]], [[["sparse_variable"], "all"]],
                         [[["sparse_fixed"], "all"]]],
    "scaled_upper_triang_masked_softmax_fusion,bias_gelu_fusion": [[True, False], [False, True]],
    "checkpoint_activations": binary,
}


@pytest.mark.parametrize("param_dict", list(parametrize(PARAMS_TO_TEST, max_tests=50, seed=None)))
def test_train(param_dict):
    @distributed_test(world_size=2)
    def wrapper():
        run_train_test(param_dict=param_dict)
    wrapper()


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
