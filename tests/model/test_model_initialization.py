import os
import re
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

if __name__ == "__main__":
    sys.path.append(os.path.abspath(''))

from megatron.neox_arguments import NeoXArgs
from megatron.global_vars import set_global_variables, get_args
from megatron.model import GPT2Model
from megatron import initialize_megatron

class ModelInitializationTest(unittest.TestCase):

    def setUp(self):
        self.neox_directory = Path(__file__).parents[2]

    def get_configs_with_path(self, configs):
        return [str(self.neox_directory / "configs" / cfg) for cfg in configs]
    
    def test_model_initialization(self):

        # intitially load config from files as would be the case in deepy.py
        yaml_list = self.get_configs_with_path(["small.yml", "local_setup.yml"])
        args_loaded = NeoXArgs.from_ymls(yaml_list)
        args_loaded.update_value("user_script", str(self.neox_directory / "pretrain_gpt2.py"))
        deepspeed_main_args = args_loaded.get_deepspeed_main_args()

        # patch sys.argv so that args can be access by set_global_variables within initialize_megatron
        with patch('sys.argv', deepspeed_main_args):
            initialize_megatron()

        # load args from global variables
        args = get_args()
        assert(isinstance(args, NeoXArgs))

        model = GPT2Model(num_tokentypes=0, parallel_output=True, inference=False, get_key_value=True)
        
        assert isinstance(model, GPT2Model)


if __name__ == "__main__":
    #test = ModelInitializationTest()
    #test.test_model_initialization()
    suite = unittest.TestSuite()
    suite.addTest(ModelInitializationTest("test_model_initialization"))
    unittest.TextTestRunner(failfast=True).run(suite)