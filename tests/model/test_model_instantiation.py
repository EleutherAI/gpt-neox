import os
import sys
import unittest
import shutil
import logging

if __name__ == "__main__":
    sys.path.append(os.path.abspath(''))

from deepspeed.runtime.pipe.engine import PipelineEngine, DeepSpeedEngine

from megatron.neox_arguments import NeoXArgs
from megatron.model import GPT2ModelPipe
from megatron import initialize_megatron
from megatron.training import setup_model_and_optimizer
from megatron.mpu import destroy_model_parallel

from tests.common import get_root_directory, iterate_all_test_configs_with_path, clear_test_dirs, TEST_CHECKPOINT_DIR, TEST_LOG_DIR, TEST_TENSORBOARD_DIR

class TestModelInstantiation(unittest.TestCase):
 
    def setUp(self):
        clear_test_dirs()

    def tearDown(self):
        clear_test_dirs()

    def run_instantiation_test(self, yaml_list):
        destroy_model_parallel() # mpu model parallel contains remaining global vars

        # intitially load config from files as would be the case in deepy.py
 
        logging.info(self.__class__.__name__ + ".run_instantiation_test() " + f"Running on: {yaml_list}")

        args_loaded = NeoXArgs.from_ymls(yaml_list)
        args_loaded.build_tokenizer()
        args_loaded.update_value("user_script", str(get_root_directory() / "pretrain_gpt2.py"))
        args_loaded.update_value("use_cpu_initialization", True)
        args_loaded.update_value("save", TEST_CHECKPOINT_DIR)
        args_loaded.update_value("load", TEST_CHECKPOINT_DIR)
        args_loaded.update_value("log_dir", TEST_LOG_DIR)
        args_loaded.update_value("tensorboard_dir", TEST_TENSORBOARD_DIR)
       
        logging.debug(self.__class__.__name__ + ".run_instantiation_test() initializing megatron")
        initialize_megatron(neox_args=args_loaded)

        logging.debug(self.__class__.__name__ + ".run_instantiation_test() initializing model")
        model, optimizer, lr_scheduler = setup_model_and_optimizer(neox_args=args_loaded, inference=False, get_key_value=True)
        
        if args_loaded.pipe_parallel_size < 2:
            self.assertTrue(isinstance(model, DeepSpeedEngine), "test model instantiation "+str(yaml_list))
        else:
            self.assertTrue(isinstance(model, PipelineEngine), "test model instantiation "+str(yaml_list))

    def test_model_instantiation(self):
        for config_list in iterate_all_test_configs_with_path():
            with self.subTest(msg="test_model_instantiation", config_list=config_list):
                self.run_instantiation_test(config_list)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestModelInstantiation("test_model_instantiation"))
    unittest.TextTestRunner(failfast=False).run(suite)