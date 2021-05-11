# Testing code to see which changes among configs break tests

import torch

from .common import TEST_CHECKPOINT_DIR, TEST_LOG_DIR, TEST_TENSORBOARD_DIR
from .common import distributed_test, get_test_configs_with_path, get_root_directory, clear_test_dirs

from .model import run_test_model_instantiation, run_train_test, run_checkpoint_test

from itertools import combinations

@distributed_test(world_size=1)
def compare_configs():
    
    #choose default params and updated ones
    base_yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_0.yml"])
    new_yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_3.yml"])
    
    from megatron.neox_arguments import NeoXArgs

    overwrite_values = {
        "user_script": str(get_root_directory() / "pretrain_gpt2.py"),
        "save": TEST_CHECKPOINT_DIR,
        "load": TEST_CHECKPOINT_DIR,
        "log_dir": TEST_LOG_DIR,
        "tensorboard_dir": TEST_TENSORBOARD_DIR,
    }
    base_args_loaded = NeoXArgs.from_ymls(base_yaml_list, overwrite_values=overwrite_values)
    new_args_loaded = NeoXArgs.from_ymls(new_yaml_list, overwrite_values=overwrite_values)

    #run_train_test(param_dict=new_args_loaded.all_config)
    
    #changed_args_loaded.all_config
    diff = {}
    for key, value in base_args_loaded.all_config.items():
        if new_args_loaded.all_config[key] != value:
            diff[key] = new_args_loaded.all_config[key]
            print(f'key: {key} original: {value}, updated: {new_args_loaded.all_config[key]}')

    # Iterate through changes and run test
    #for key, value in diff.items():
    #    param_dict = base_args_loaded.all_config
    #    param_dict[key] = value
    #    print(f'running setup = key: {key} original: {base_args_loaded.all_config[key]}, updated: {value}')
    #
    #    # Change desired test here
    #    run_train_test(param_dict=param_dict)
    #
    #    print(f'run setup = key: {key} original: {base_args_loaded.all_config[key]}, updated: {value}')

    # Iterate through pair's of changes to find what causes issue
    perms = list(combinations(diff.items(), 2))
    for items in perms:
        param_dict = base_args_loaded.all_config
        print('running setup with:')
        for item in items:
            param_dict[item[0]] = item[1]
            print(f'key: {item[0]} original: {base_args_loaded.all_config[item[0]]}, updated: {item[1]}')

        run_train_test(param_dict=param_dict)

        print('finished running setup with:')
        for item in items:
            print(f'key: {item[0]} original: {base_args_loaded.all_config[item[0]]}, updated: {item[1]}')