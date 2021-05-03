import os
from pathlib import Path

from ..common import TEST_CHECKPOINT_DIR, TEST_LOG_DIR, TEST_TENSORBOARD_DIR
from ..common import distributed_test, get_root_directory, get_test_configs_with_path, clear_test_dirs

import torch

@distributed_test(world_size=1)
def test_model_checkpoint_small():
    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small.yml"])
    run_checkpoint_test(yaml_list)

# @distributed_test(world_size=2)
# def test_model_checkpoint_small_pp():
#     yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_pp.yml"])
#     run_checkpoint_test(yaml_list)

def run_checkpoint_test(yaml_list):
    from megatron.neox_arguments import NeoXArgs
    from megatron import initialize_megatron
    from megatron.text_generation_utils import get_batch, forward_model
    from megatron.training import setup_model_and_optimizer
    from megatron.mpu import destroy_model_parallel

    from megatron.checkpointing import load_checkpoint
    from megatron.checkpointing import save_checkpoint

    destroy_model_parallel() # mpu model parallel contains remaining global vars
    clear_test_dirs()

    # intitially load config from files as would be the case in deepy.py
    args_loaded = NeoXArgs.from_ymls(yaml_list)
    args_loaded.build_tokenizer()
    args_loaded.update_value("user_script", str(get_root_directory() / "pretrain_gpt2.py"))
    args_loaded.update_value("use_cpu_initialization", True)
    args_loaded.update_value("save", TEST_CHECKPOINT_DIR)
    args_loaded.update_value("load", TEST_CHECKPOINT_DIR)
    args_loaded.update_value("log_dir", TEST_LOG_DIR)
    args_loaded.update_value("tensorboard_dir", TEST_TENSORBOARD_DIR)
    
    initialize_megatron(neox_args=args_loaded)

    model, optimizer, lr_scheduler = setup_model_and_optimizer(neox_args=args_loaded, inference=False, get_key_value=True)
    model.eval()

    # save model checkpoint
    save_checkpoint(neox_args=args_loaded, iteration=42, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
    

    # reload model from checkpoint
    args_reloaded = NeoXArgs.from_ymls(yaml_list)
    args_reloaded.build_tokenizer()
    args_reloaded.update_value("user_script", str(get_root_directory() / "pretrain_gpt2.py"))
    args_reloaded.update_value("use_cpu_initialization", True)
    args_reloaded.update_value("save", TEST_CHECKPOINT_DIR)
    args_reloaded.update_value("load", TEST_CHECKPOINT_DIR)
    args_reloaded.update_value("log_dir", TEST_LOG_DIR)
    args_reloaded.update_value("tensorboard_dir", TEST_TENSORBOARD_DIR)

    raise ValueError("TEST_CHECKPOINT_DIR: "+str(TEST_CHECKPOINT_DIR))

    reloaded_model, optimizer, lr_scheduler = setup_model_and_optimizer(neox_args=args_reloaded, inference=False, get_key_value=True)
    iteration = load_checkpoint(neox_args=args_reloaded, model=reloaded_model, optimizer=optimizer, lr_scheduler=lr_scheduler)
    reloaded_model.eval()

    #ensure same checkpoint is loaded
    assert iteration == 42, "run_checkpoint_test() iteration loaded from checkpoint correct"

    #check all weight groups are the same
    for idx, ((n1, p1), (n2, p2)) in enumerate(zip(list(model.module.named_parameters()), list(reloaded_model.module.named_parameters()))):
        assert n1 == n2
        params_equal = (p1 == p2).all().item()
        assert params_equal, "run_checkpoint_test() params equal: "+str(n1)

    clear_test_dirs()