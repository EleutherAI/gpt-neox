import os
from pathlib import Path

from ..common import TEST_CHECKPOINT_DIR, TEST_LOG_DIR, TEST_TENSORBOARD_DIR
from ..common import distributed_test, get_root_directory, get_test_configs_with_path, clear_test_dirs

import torch

@distributed_test(world_size=1)
def test_model_checkpoint_small():
    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small.yml"])
    run_checkpoint_test(yaml_list, do_forward_pass=False)

@distributed_test(world_size=2)
def test_model_checkpoint_small_pp():
    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_pp.yml"])
    run_checkpoint_test(yaml_list, do_forward_pass=False)

def run_checkpoint_test(yaml_list, do_forward_pass=False):
    from megatron.neox_arguments import NeoXArgs
    from megatron import initialize_megatron
    from megatron.text_generation_utils import get_batch, forward_model
    from megatron.training import setup_model_and_optimizer
    from megatron.mpu import destroy_model_parallel

    from megatron.checkpointing import load_checkpoint
    from megatron.checkpointing import save_checkpoint

    destroy_model_parallel() # mpu model parallel contains remaining global vars

    # intitially load config from files as would be the case in deepy.py
    args_loaded = NeoXArgs.from_ymls(yaml_list)
    args_loaded.build_tokenizer()
    args_loaded.update_value("user_script", str(get_root_directory() / "pretrain_gpt2.py"))
    args_loaded.update_value("use_cpu_initialization", True)
    args_loaded.update_value("save", TEST_CHECKPOINT_DIR)
    args_loaded.update_value("load", TEST_CHECKPOINT_DIR)
    args_loaded.update_value("log_dir", TEST_LOG_DIR)
    args_loaded.update_value("tensorboard_dir", TEST_TENSORBOARD_DIR)
    if torch.distributed.get_world_size() == 1 or torch.distributed.get_rank() == 0:
        clear_test_dirs()
    
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
    args_reloaded = NeoXArgs.from_ymls(yaml_list)
    args_reloaded.build_tokenizer()
    args_reloaded.update_value("user_script", str(get_root_directory() / "pretrain_gpt2.py"))
    args_reloaded.update_value("use_cpu_initialization", True)
    args_reloaded.update_value("save", TEST_CHECKPOINT_DIR)
    args_reloaded.update_value("load", TEST_CHECKPOINT_DIR)
    args_reloaded.update_value("log_dir", TEST_LOG_DIR)
    args_reloaded.update_value("tensorboard_dir", TEST_TENSORBOARD_DIR)

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