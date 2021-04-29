import os
import sys
import unittest
from unittest.mock import patch

if __name__ == "__main__":
    sys.path.append(os.path.abspath(''))

from megatron.neox_arguments import NeoXArgs
from megatron.global_vars import get_args, reset_global_variables
from megatron.model import GPT2ModelPipe
from megatron import initialize_megatron
from megatron import mpu

from tests.common import get_root_directory, get_configs_with_path

class TestModelInitialization(unittest.TestCase):
 
    def test_model_initialization(self):
        reset_global_variables()

        # intitially load config from files as would be the case in deepy.py
        yaml_list = get_configs_with_path(["small.yml", "local_setup.yml"])
        args_loaded = NeoXArgs.from_ymls(yaml_list)
        args_loaded.update_value("user_script", str(get_root_directory() / "pretrain_gpt2.py"))
        args_loaded.update_value("pipe_parallel_size", 1) # overwrite pipeline parameter, config in small.yml may have changed!
        deepspeed_main_args = args_loaded.get_deepspeed_main_args()

        # patch sys.argv so that args can be access by set_global_variables within initialize_megatron
        with patch('sys.argv', deepspeed_main_args):
            initialize_megatron()

        # load args from global variables
        args = get_args() # TODO remove_global_vars
        self.assertTrue(isinstance(args, NeoXArgs))

        print("topology", mpu.get_topology(), flush=True)
        model = GPT2ModelPipe(num_tokentypes=0, parallel_output=True, topology=mpu.get_topology(), inference=False, get_key_value=True)
        
        self.assertTrue(isinstance(model, GPT2ModelPipe)) 


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestModelInitializationPipeline("test_model_initialization_pipeline"))
    unittest.TextTestRunner(failfast=True).run(suite)