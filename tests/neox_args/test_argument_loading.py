import os
import sys
import unittest
from pathlib import Path

from megatron.neox_arguments import NeoXArgs

class LoadArgumentTest(unittest.TestCase):

    def setUp(self):
        self.config_directory = Path(__file__).parents[2] / "configs"

    def get_configs_with_path(self, configs):
        return [str(self.config_directory / cfg) for cfg in configs]

    def test_load_arguments_small_local_setup(self):
        yaml_list = self.get_configs_with_path(["small.yml", "local_setup.yml"])
        args_loaded = NeoXArgs.from_ymls(yaml_list)
        assert(isinstance(args_loaded, NeoXArgs))
        
    def test_load_arguments_medium_local_setup(self):
        yaml_list = self.get_configs_with_path(["medium.yml", "local_setup.yml"])
        args_loaded = NeoXArgs.from_ymls(yaml_list)
        assert(isinstance(args_loaded, NeoXArgs))

    def test_load_arguments_large_local_setup(self):
        yaml_list = self.get_configs_with_path(["large.yml", "local_setup.yml"])
        args_loaded = NeoXArgs.from_ymls(yaml_list)
        assert(isinstance(args_loaded, NeoXArgs))

    def test_loal_local_setup(self):
        yaml_list = self.get_configs_with_path(["local_setup.yml"])
        self.assertRaises(AssertionError, NeoXArgs, (yaml_list, ))
        