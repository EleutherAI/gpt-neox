"""
instantiate models, save checkpoints, load checkpoints, compare loaded parameters to saved parameters and compare forward pass outputs

This tests contain a relatively large number of functions. They are not split into separate tests because a lot of boilerplate (e.g. instantiate model) needs
to run in order to perform follow up tests. Joining in one test reduces runtime at the expense of decreased transparency of test results in case of failures.
"""
import json
import os
from pathlib import Path

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.abspath(''))

from tests.common import TEST_CHECKPOINT_DIR, TEST_LOG_DIR, TEST_TENSORBOARD_DIR
from tests.common import distributed_test, get_root_directory, get_test_configs_with_path, clear_test_dirs

import torch

@distributed_test(world_size=1)
def test_model_generation_unconditional_small_0_greedy():
    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_0.yml", "text_generation_greedy.yml"])
    run_generate_uncondional_test(yaml_list, greedy=True)

@distributed_test(world_size=1)
def test_model_generation_unconditional_small_0_0():
    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_0.yml", "text_generation_0.yml"])
    run_generate_uncondional_test(yaml_list, greedy=False)

@distributed_test(world_size=1)
def test_model_generation_unconditional_small_0_1():
    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_0.yml", "text_generation_1.yml"])
    run_generate_uncondional_test(yaml_list, greedy=False)

@distributed_test(world_size=1)
def test_model_generation_unconditional_small_0_2():
    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_0.yml", "text_generation_2.yml"])
    run_generate_uncondional_test(yaml_list, greedy=False)

@distributed_test(world_size=1)
def test_model_generation_unconditional_small_1():
    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_1.yml", "text_generation_greedy.yml"])
    run_generate_uncondional_test(yaml_list, greedy=True)

# @distributed_test(world_size=2)
# def test_model_generation_unconditional_small_2():
#     yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_2.yml", "text_generation_greedy.yml"])
#     run_generate_uncondional_test(yaml_list, greedy=True)

@distributed_test(world_size=1)
def test_model_generation_unconditional_small_3():
    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_3.yml", "text_generation_greedy.yml"])
    run_generate_uncondional_test(yaml_list, greedy=True)

@distributed_test(world_size=2)
def test_model_generation_unconditional_small_4():
    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_4.yml", "text_generation_greedy.yml"])
    run_generate_uncondional_test(yaml_list, greedy=True)


@distributed_test(world_size=1)
def test_model_generation_input_file_small_0():
    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_0.yml"])
    run_generate_input_file_test(yaml_list)

def run_generate_uncondional_test(yaml_list, greedy=False):
    from megatron.neox_arguments import NeoXArgs
    from megatron import initialize_megatron
    from megatron.training import setup_model_and_optimizer
    from megatron.mpu import destroy_model_parallel
    from megatron.utils import is_mp_rank_0

    from megatron.text_generation_utils import generate_samples_unconditional

    sample_output_file = "test_sample_output.txt"
    num_samples = 3

    destroy_model_parallel() # mpu model parallel contains remaining global vars

    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1 or torch.distributed.get_rank() == 0:
        clear_test_dirs()

    # intitially load config from files as would be the case in deepy.py
    args_loaded = NeoXArgs.from_ymls(yaml_list, overwrite_values={
        "user_script": str(get_root_directory() / "text_gen_gpt2.py"),
        "save": TEST_CHECKPOINT_DIR,
        "load": TEST_CHECKPOINT_DIR,
        "log_dir": TEST_LOG_DIR,
        "tensorboard_dir": TEST_TENSORBOARD_DIR,

        "checkpoint_activations": False,
        "partition_activations": False,
        "no_load_optim": False, #TODO True == deepseed tries to initialize!?

        "text_gen_type": "unconditional",
        "sample_output_file": sample_output_file,
        "num_samples": num_samples,
    })
    args_loaded.build_tokenizer()
    
    initialize_megatron(neox_args=args_loaded)

    model, _, _ = setup_model_and_optimizer(neox_args=args_loaded, inference=True, get_key_value=True)
    model.eval()

    generate_samples_unconditional(
            neox_args=args_loaded, 
            model=model,
            number_of_samples=args_loaded.num_samples,
            output_file=args_loaded.sample_output_file,
            maximum_tokens = args_loaded.maximum_tokens, 
            recompute = len(args_loaded.sparsity_config) > 0, 
            temperature = args_loaded.temperature,
            top_k = args_loaded.top_k, 
            top_p = args_loaded.top_p
    )
    torch.distributed.barrier() # torch distributed barrier is necessary here, otherwise process writing output file may not be done when calling the next checks

    assert Path(sample_output_file).is_file(), "unconditional samples generated"
    
    sample_count = 0
    last_sample = None
    with open(sample_output_file, "r") as f:
        for sample_src in f:
            if sample_src == "": continue
            sample_count += 1
            loaded = json.loads(sample_src)
            if last_sample is not None:
                if greedy:
                    assert last_sample == loaded["text"], "unconditional greedy generation always returning same sample"
                else:
                    assert last_sample != loaded["text"], "unconditional generation (non greedy) never returning same sample twice"
            last_sample = loaded["text"]

    assert sample_count == num_samples, "generated the right number of unconditional samples"

    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1 or torch.distributed.get_rank() == 0:
        Path(sample_output_file).unlink()

def run_generate_input_file_test(yaml_list):
    from megatron.neox_arguments import NeoXArgs
    from megatron import initialize_megatron
    from megatron.training import setup_model_and_optimizer
    from megatron.mpu import destroy_model_parallel

    from megatron.text_generation_utils import generate_samples_input_from_file

    sample_input_file = "test_sample_input.txt"
    sample_output_file = "test_sample_output.txt"

    destroy_model_parallel() # mpu model parallel contains remaining global vars

    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1 or torch.distributed.get_rank() == 0:
        clear_test_dirs()

    with open(sample_input_file, "w") as f:
        f.write("This is a prompt\nThis too")

    # intitially load config from files as would be the case in deepy.py
    args_loaded = NeoXArgs.from_ymls(yaml_list, overwrite_values={
        "user_script": str(get_root_directory() / "text_gen_gpt2.py"),
        "save": TEST_CHECKPOINT_DIR,
        "load": TEST_CHECKPOINT_DIR,
        "log_dir": TEST_LOG_DIR,
        "tensorboard_dir": TEST_TENSORBOARD_DIR,

        "checkpoint_activations": False,
        "partition_activations": False,
        "no_load_optim": True,

        "text_gen_type": "input-file",
        "sample_input_file": sample_input_file,
        "sample_output_file": sample_output_file
    })
    args_loaded.build_tokenizer()
    
    initialize_megatron(neox_args=args_loaded)

    model, _, _ = setup_model_and_optimizer(neox_args=args_loaded, inference=True, get_key_value=True)
    model.eval()

    generate_samples_input_from_file(
            neox_args=args_loaded, 
            model=model,
            input_file=args_loaded.sample_input_file,
            output_file=args_loaded.sample_output_file,
            maximum_tokens = args_loaded.maximum_tokens, 
            recompute = len(args_loaded.sparsity_config) > 0, 
            temperature = args_loaded.temperature,
            top_k = args_loaded.top_k, 
            top_p = args_loaded.top_p
    )
    torch.distributed.barrier() # torch distributed barrier is necessary here, otherwise process writing output file may not be done when calling the next checks

    assert Path(sample_output_file).is_file(), "unconditional samples generated"

    sample_count = 0
    with open(sample_output_file, "r") as f:
        for sample_src in f:
            if sample_src == "": continue
            sample_count += 1
            loaded = json.loads(sample_src)

    assert sample_count == 2, "generated the right number of unconditional samples"

    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1 or torch.distributed.get_rank() == 0:
        Path(sample_input_file).unlink()
        Path(sample_output_file).unlink()


if __name__ == "__main__":
    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_0.yml"])
    run_generate_uncondional_test(yaml_list)