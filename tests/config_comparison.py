# Should just be called using: "python tests/config_comparison.py"
# Testing code to see which changes among configs break tests

# Hacky can't remember how to move back to root dir
import sys  
from pathlib import Path  
file = Path(__file__).resolve()  
package_root_directory = file.parents[1]  
sys.path.append(str(package_root_directory))  

from itertools import combinations
from tests.model import run_test_model_instantiation, run_train_test, run_checkpoint_test
from tests.common import TEST_CHECKPOINT_DIR, TEST_LOG_DIR, TEST_TENSORBOARD_DIR
from tests.common import distributed_test, get_test_configs_with_path, get_root_directory, clear_test_dirs

# World size might need to be adjusted depending on test
@distributed_test(world_size=1)
def main(subsequence_length: int = 2):
    """Allows you to easily compare sets of combinations to find which changes are causing issues
    Args:
        subsequence_length (int, optional): the length of subsequences of elements from the input iterable. Defaults to 2.
    """

    #choose default params and updated ones
    base_yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_0.yml"])
    new_yaml_list = get_test_configs_with_path(["test_local_setup.yml", "test_small_3.yml"])

    # Need to import here as distributed
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

    # Find difference between configs
    diff = {}
    for key, value in base_args_loaded.all_config.items():
        if new_args_loaded.all_config[key] != value:
            diff[key] = new_args_loaded.all_config[key]
            print(f'key: {key} original: {value}, updated: {new_args_loaded.all_config[key]}')


    perms = list(combinations(diff.items(), subsequence_length))

    # Iterate over combinations and run the test function
    # and print information so you can debug from console as program is distributed
    for items in perms:
        param_dict = base_args_loaded.all_config
        print('running setup with:')
        for item in items:
            param_dict[item[0]] = item[1]
            print(f'key: {item[0]} original: {base_args_loaded.all_config[item[0]]}, updated: {item[1]}')

        # These are interchangable

        run_train_test(param_dict=param_dict)
        #run_test_model_instantiation(param_dict=param_dict)
        #run_checkpoint_test(param_dict=param_dict)

        print('finished running setup with:')
        for item in items:
            print(f'key: {item[0]} original: {base_args_loaded.all_config[item[0]]}, updated: {item[1]}')

if __name__ == '__main__':
    main()