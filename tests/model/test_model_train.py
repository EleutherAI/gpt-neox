"""
instantiate models, save checkpoints, load checkpoints, compare loaded parameters to saved parameters and compare forward pass outputs

This tests contain a relatively large number of functions. They are not split into separate tests because a lot of boilerplate (e.g. instantiate model) needs
to run in order to perform follow up tests. Joining in one test reduces runtime at the expense of decreased transparency of test results in case of failures.
"""

import os
from pathlib import Path

from ..common import TEST_CHECKPOINT_DIR, TEST_LOG_DIR, TEST_TENSORBOARD_DIR
from ..common import distributed_test, get_root_directory, get_test_configs_with_path, clear_test_dirs

import torch

@distributed_test(world_size=1)
def test_model_train_small_0():
    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_0.yml"])
    run_train_test(yaml_list=yaml_list)

@distributed_test(world_size=1)
def test_model_train_small_1():
    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_1.yml"])
    run_train_test(yaml_list=yaml_list)

# TODO after sorting out scaled-upper-triang-masked-softmax-fusion and fp16
#@distributed_test(world_size=2)
#def test_model_train_small_2():
#    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_2.yml"])
#    run_train_test(yaml_list=yaml_list)

# TODO after sorting out RPE + sparse attention
#@distributed_test(world_size=1)
#def test_model_train_small_3():
#    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_3.yml"])
#    run_train_test(yaml_list=yaml_list)

@distributed_test(world_size=2)
def test_model_train_small_4():
    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_4.yml"])
    run_train_test(yaml_list=yaml_list)

def run_train_test(yaml_list=None, param_dict=None):

    from megatron.neox_arguments import NeoXArgs
    from megatron import initialize_megatron
    from megatron.training import setup_model_and_optimizer, train_step
    from megatron.mpu import destroy_model_parallel
    from megatron.utils import Timers

    max_steps = 256 


    destroy_model_parallel() # mpu model parallel contains remaining global vars

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

    # intitially load config from files as would be the case in deepy.py
    if yaml_list is not None:
        args_loaded = NeoXArgs.from_ymls(yaml_list, overwrite_values=overwrite_values)
    else:
        p_dict = param_dict.copy()
        p_dict.update(overwrite_values)
        args_loaded = NeoXArgs.from_dict(p_dict)
    
    args_loaded.build_tokenizer()
    
    initialize_megatron(neox_args=args_loaded)

    model, optimizer, lr_scheduler = setup_model_and_optimizer(neox_args=args_loaded, inference=False, get_key_value=True)
    model.train()

 
    timers = Timers(use_wandb=False, tensorboard_writer=None)
    
    # generate some random data on which we can overfit
    # context size of data is model seq_len + 1 in order to compute loss
    data_list = list()
    context_tokens_tensor = torch.randint(0, args_loaded.padded_vocab_size, (4, args_loaded.seq_length + 1 )).to(torch.int64) 
    for i in range(max_steps):
        data_list.append({ "text": context_tokens_tensor.clone() })
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
                return # all good
    
    # loss should have decreased by now (otherwise increasing the max_steps parameter could have the testcase pass)
    assert losses[-1] < losses[-2], "run_train_test() loss going down within "+str(max_steps)+" steps"

    if torch.distributed.get_world_size() == 1 or torch.distributed.get_rank() == 0:
       clear_test_dirs()