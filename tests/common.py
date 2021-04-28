"""
collection of reusable functions in the context of testing
"""

from pathlib import Path

def get_root_directory():
    return Path(__file__).parents[1]

def get_config_directory():
    return get_root_directory() / "configs"

def get_configs_with_path(configs):
    return [str(get_config_directory() / cfg) for cfg in configs]
