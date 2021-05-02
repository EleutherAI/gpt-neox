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

from tests.common import get_root_directory, get_configs_with_path, get_test_configs_with_path, clear_test_dirs, TEST_CHECKPOINT_DIR, TEST_LOG_DIR

class TestModelInstantiation(unittest.TestCase):
 
    def setUp(self):
        clear_test_dirs()

    def tearDown(self):
        clear_test_dirs()

    def run_instantiation_test(self, yaml_list, model_class_expected):
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
       
        logging.debug(self.__class__.__name__ + ".run_instantiation_test() initializing megatron")
        initialize_megatron(neox_args=args_loaded)

        logging.debug(self.__class__.__name__ + ".run_instantiation_test() initializing model")
        model, optimizer, lr_scheduler = setup_model_and_optimizer(neox_args=args_loaded, inference=False, get_key_value=True)
        
        self.assertTrue(isinstance(model, model_class_expected)) 

    def test_model_instantiation_small(self):
        self.run_instantiation_test(get_configs_with_path(["local_setup.yml", "small.yml"]), PipelineEngine)

    def test_model_instantiation_medium(self):
        self.run_instantiation_test(get_configs_with_path(["local_setup.yml", "medium.yml"]), PipelineEngine)
        
    def test_model_instantiation_small_test(self):
        self.run_instantiation_test(get_test_configs_with_path(["test_local_setup.yml", "test_small.yml"]), DeepSpeedEngine)

    def test_model_instantiation_medium_test(self):
        self.run_instantiation_test(get_test_configs_with_path(["test_local_setup.yml", "test_medium.yml"]), DeepSpeedEngine)

    def test_model_instantiation_small_sparse_test(self):
        self.run_instantiation_test(get_test_configs_with_path(["test_local_setup.yml", "test_small.yml", "test_sparse.yml"]), DeepSpeedEngine)

if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestModelInstantiation("test_model_instantiation_small_sparse_test"))
    unittest.TextTestRunner(failfast=True).run(suite)