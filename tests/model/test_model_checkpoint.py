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
def test_model_checkpoint_small_0():
    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_0.yml"])
    run_checkpoint_test(yaml_list=yaml_list, do_forward_pass=False, cpu=True)

@distributed_test(world_size=1)
def test_model_checkpoint_small_1():
    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_1.yml"])
    run_checkpoint_test(yaml_list=yaml_list, do_forward_pass=False, cpu=True)

@distributed_test(world_size=2)
def test_model_checkpoint_small_2():
    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_2.yml"])
    run_checkpoint_test(yaml_list=yaml_list, do_forward_pass=False, cpu=False)

@distributed_test(world_size=1)
def test_model_checkpoint_small_3():
    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_3.yml"])
    run_checkpoint_test(yaml_list=yaml_list, do_forward_pass=False, cpu=False)

@distributed_test(world_size=2)
def test_model_checkpoint_small_4():
    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_4.yml"])
    run_checkpoint_test(yaml_list=yaml_list, do_forward_pass=False, cpu=False)

def run_checkpoint_test(yaml_list=None, param_dict=None, do_forward_pass=False, cpu=False):
    from megatron.neox_arguments import NeoXArgs
    from megatron import initialize_megatron
    from megatron.text_generation_utils import get_batch, forward_model
    from megatron.training import setup_model_and_optimizer
    from megatron.mpu import destroy_model_parallel

    from megatron.checkpointing import load_checkpoint
    from megatron.checkpointing import save_checkpoint

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

    model, optimizer, lr_scheduler = setup_model_and_optimizer(neox_args=args_loaded, inference=True, get_key_value=True)
    model.eval()

    # save model checkpoint
    save_checkpoint(neox_args=args_loaded, iteration=42, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
    
    
    # forward
    if do_forward_pass:
        context_tokens_tensor = torch.cuda.LongTensor([[1,2,3,4,5],[1,2,3,4,5],[6,7,8,9,10],[1,2,3,4,100]])
        tokens, attention_mask, position_ids = get_batch(args_loaded, context_tokens_tensor)
        logits, layer_past = forward_model(args_loaded, model, (tokens, position_ids, attention_mask, torch.Tensor()))
        

        # assert logits are the right shape
        assert torch.is_tensor(logits), "run_checkpoint_test() forward output is tensor"
        assert logits.size(0) == context_tokens_tensor.size(0), "run_checkpoint_test() batch size correct"
        assert logits.size(1) == context_tokens_tensor.size(1), "run_checkpoint_test() context size correct"
    
        # assert correct behaviour
        assert torch.isclose(logits[0], logits[1]).all().item(), "run_checkpoint_test() forward independent of batch index"
        assert not torch.isclose(logits[1], logits[2]).all().item(), "run_checkpoint_test() forward produced different outputs for different inputs"
        assert torch.isclose(logits[1, 3], logits[3, 3]).all().item(), "run_checkpoint_test() forward masks right side tokens"

    # reload model from checkpoint
    if yaml_list is not None:
        args_reloaded = NeoXArgs.from_ymls(yaml_list, overwrite_values=overwrite_values)
    else:
        p_dict = param_dict.copy()
        p_dict.update(overwrite_values)
        args_reloaded = NeoXArgs.from_dict(p_dict)

    args_reloaded.build_tokenizer()

    reloaded_model, optimizer, lr_scheduler = setup_model_and_optimizer(neox_args=args_reloaded, inference=True, get_key_value=True)
    iteration = load_checkpoint(neox_args=args_reloaded, model=reloaded_model, optimizer=optimizer, lr_scheduler=lr_scheduler)
    reloaded_model.eval()

    #ensure same checkpoint is loaded
    assert iteration == 42, "run_checkpoint_test() iteration loaded from checkpoint correct"

    if do_forward_pass:
        #check re-loaded model returns the same results
        logits_reloaded, layer_past = forward_model(args_reloaded, model, (tokens, position_ids, attention_mask))
        assert torch.isclose(logits, logits_reloaded).all().item(), "run_checkpoint_test() forward output after reloading checkpoint unchanged"

    #check all weight groups are the same
    for idx, ((n1, p1), (n2, p2)) in enumerate(zip(list(model.module.named_parameters()), list(reloaded_model.module.named_parameters()))):
        assert n1 == n2
        params_equal = (p1 == p2).all().item()
        assert params_equal, "run_checkpoint_test() params equal: "+str(n1)

    if torch.distributed.get_world_size() == 1 or torch.distributed.get_rank() == 0:
        clear_test_dirs()