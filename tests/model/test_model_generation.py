"""
instantiate models, save checkpoints, load checkpoints, compare loaded parameters to saved parameters and compare forward pass outputs

This tests contain a relatively large number of functions. They are not split into separate tests because a lot of boilerplate (e.g. instantiate model) needs
to run in order to perform follow up tests. Joining in one test reduces runtime at the expense of decreased transparency of test results in case of failures.
"""
import json
import os
from pathlib import Path

from ..common import TEST_CHECKPOINT_DIR, TEST_LOG_DIR, TEST_TENSORBOARD_DIR
from ..common import distributed_test, get_root_directory, get_test_configs_with_path, clear_test_dirs

import torch

@distributed_test(world_size=1)
def test_model_generation_unconditional_small_0():
    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_0.yml"])
    run_generate_uncondional_test(yaml_list)

# @distributed_test(world_size=1)
# def test_model_generation_unconditional_small_1():
#     yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_1.yml"])
#     run_generate_uncondional_test(yaml_list)

# # for some reason this testcase is running way to long
# # potentially the optimizer problem?
# # @distributed_test(world_size=2)
# # def test_model_generation_unconditional_small_2():
# #     yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_2.yml"])
# #     run_generate_uncondional_test(yaml_list)

# @distributed_test(world_size=1)
# def test_model_generation_unconditional_small_3():
#     yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_3.yml"])
#     run_generate_uncondional_test(yaml_list)

# @distributed_test(world_size=2)
# def test_model_generation_unconditional_small_4():
#     yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_4.yml"])
#     run_generate_uncondional_test(yaml_list)


# @distributed_test(world_size=1)
# def test_model_generation_input_file_small_0():
#     yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_0.yml"])
#     run_generate_input_file_test(yaml_list)

# @distributed_test(world_size=1)
# def test_model_generation_input_file_small_1():
#     yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_1.yml"])
#     run_generate_input_file_test(yaml_list)

# # for some reason this testcase is running way to long
# # potentially the optimizer problem?
# # @distributed_test(world_size=2)
# # def test_model_generation_input_file_small_2():
# #     yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_2.yml"])
# #     run_generate_input_file_test(yaml_list)

# @distributed_test(world_size=1)
# def test_model_generation_input_file_small_3():
#     yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_3.yml"])
#     run_generate_input_file_test(yaml_list)

# @distributed_test(world_size=2)
# def test_model_generation_input_file_small_4():
#     yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_4.yml"])
#     run_generate_input_file_test(yaml_list)


def run_generate_uncondional_test(yaml_list):
    from megatron.neox_arguments import NeoXArgs
    from megatron import initialize_megatron
    from megatron.training import setup_model_and_optimizer
    from megatron.mpu import destroy_model_parallel

    from megatron.text_generation_utils import generate_and_write_samples_unconditional

    genfile = "test_generation_file.txt"
    num_samples = 3

    destroy_model_parallel() # mpu model parallel contains remaining global vars

    if torch.distributed.get_world_size() == 1 or torch.distributed.get_rank() == 0:
        clear_test_dirs()

    # intitially load config from files as would be the case in deepy.py
    args_loaded = NeoXArgs.from_ymls(yaml_list, overwrite_values={
        "user_script": str(get_root_directory() / "pretrain_gpt2.py"),
        "save": TEST_CHECKPOINT_DIR,
        "load": TEST_CHECKPOINT_DIR,
        "log_dir": TEST_LOG_DIR,
        "tensorboard_dir": TEST_TENSORBOARD_DIR,

        "checkpoint_activations": False,
        "partition_activations": False,
        "no_load_optim": True,

        "text_gen_type": "unconditional",
        "genfile": genfile,
        "num_samples": num_samples,
    })
    args_loaded.build_tokenizer()
    
    initialize_megatron(neox_args=args_loaded)

    model, _, _ = setup_model_and_optimizer(neox_args=args_loaded, inference=True, get_key_value=True)
    model.eval()

    generate_and_write_samples_unconditional(neox_args=args_loaded, model=model)

    assert Path(genfile).is_file(), "unconditional samples generated"

    sample_count = 0
    with open(genfile, "r") as f:
        for sample_src in f:
            if sample_src == "": continue
            sample_count += 1
            loaded = json.loads(sample_src)

    assert sample_count == num_samples, "generated the right number of unconditional samples"

    Path(genfile).unlink()

def run_generate_input_file_test(yaml_list):
    from megatron.neox_arguments import NeoXArgs
    from megatron import initialize_megatron
    from megatron.training import setup_model_and_optimizer
    from megatron.mpu import destroy_model_parallel

    from megatron.text_generation_utils import generate_samples_input_from_file

    sample_input_file = "test_generation_input.txt"
    sample_output_file = "test_generation_output.txt"
    num_samples = 3

    with open(sample_input_file, "w") as f:
        f.write("This is the first prompt")
    
    destroy_model_parallel() # mpu model parallel contains remaining global vars

    if torch.distributed.get_world_size() == 1 or torch.distributed.get_rank() == 0:
        clear_test_dirs()

    # intitially load config from files as would be the case in deepy.py
    args_loaded = NeoXArgs.from_ymls(yaml_list, overwrite_values={
        "user_script": str(get_root_directory() / "pretrain_gpt2.py"),
        "save": TEST_CHECKPOINT_DIR,
        "load": TEST_CHECKPOINT_DIR,
        "log_dir": TEST_LOG_DIR,
        "tensorboard_dir": TEST_TENSORBOARD_DIR,

        "checkpoint_activations": False,
        "partition_activations": False,
        "no_load_optim": True,

        "text_gen_type": "input-file",
        "sample_input_file": sample_input_file,
        "sample_output_file": sample_output_file,
        "num_samples": num_samples,
    })
    args_loaded.build_tokenizer()
    
    initialize_megatron(neox_args=args_loaded)

    model, _, _ = setup_model_and_optimizer(neox_args=args_loaded, inference=True, get_key_value=True)
    model.eval()

    generate_samples_input_from_file(neox_args=args_loaded, model=model)

    assert Path(genfile).is_file(), "unconditional samples generated"

    sample_count = 0
    with open(sample_output_file, "r") as f:
        for sample_src in f:
            if sample_src == "": continue
            sample_count += 1
            loaded = json.loads(sample_src)

    assert sample_count == 2 * num_samples, "generated the right number of unconditional samples"

    #Path(sample_input_file).unlink()
    #Path(sample_output_file).unlink()


if __name__ == "__main__":
    yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_0.yml"])
    run_generate_input_file_test(yaml_list)