"""
collection of reusable functions in the context of testing
"""

import os
import shutil
from pathlib import Path

TEST_CHECKPOINT_DIR = "test_checkpoint"
TEST_LOG_DIR = "test_logs"

def get_root_directory():
    return Path(__file__).parents[1]

def get_config_directory():
    return get_root_directory() / "configs"

def get_configs_with_path(configs):
    return [str(get_config_directory() / cfg) for cfg in configs]

def get_test_configs_with_path(configs):
    test_config_dir = Path(__file__).parent / "model" / "test_configs"
    return [str((test_config_dir / cfg).absolute()) for cfg in configs]

def clear_test_dirs():
    log_dir = os.path.join(get_root_directory(),TEST_LOG_DIR)
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)

    checkpoint_dir = os.path.join(get_root_directory(), TEST_CHECKPOINT_DIR)
    if os.path.isdir(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)