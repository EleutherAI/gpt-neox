import os
import sys
import shutil
import unittest
import logging
from pathlib import Path

if __name__ == "__main__":
    sys.path.append(os.path.abspath(''))

from megatron.neox_arguments import NeoXArgs
from megatron import initialize_megatron
from megatron.text_generation_utils import get_batch, forward_model
from megatron.training import setup_model_and_optimizer
from megatron.mpu import destroy_model_parallel

from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint

from tests.common import get_root_directory, get_configs_with_path, get_test_configs_with_path, clear_test_dirs, TEST_CHECKPOINT_DIR, TEST_LOG_DIR

import torch

class TestModelCheckpoint(unittest.TestCase):

    def setUp(self):
        clear_test_dirs()

    def tearDown(self):
        clear_test_dirs()

    def run_checkpoint_test(self, yaml_list):
        destroy_model_parallel() # mpu model parallel contains remaining global vars

        # intitially load config from files as would be the case in deepy.py
 
        logging.info(self.__class__.__name__ + ".run_checkpoint_test() " + f"Running on: {yaml_list}")

        args_loaded = NeoXArgs.from_ymls(yaml_list)
        args_loaded.build_tokenizer()
        args_loaded.update_value("user_script", str(get_root_directory() / "pretrain_gpt2.py"))
        args_loaded.update_value("use_cpu_initialization", True)
        args_loaded.update_value("save", TEST_CHECKPOINT_DIR)
        args_loaded.update_value("load", TEST_CHECKPOINT_DIR)
        args_loaded.update_value("log_dir", TEST_LOG_DIR)
       
        logging.debug(self.__class__.__name__ + ".run_checkpoint_test() initializing megatron")
        initialize_megatron(neox_args=args_loaded)

        logging.debug(self.__class__.__name__ + ".run_checkpoint_test() initializing model")
        model, optimizer, lr_scheduler = setup_model_and_optimizer(neox_args=args_loaded, inference=False, get_key_value=True)
        model.eval()

        # save model checkpoint
        logging.debug( self.__class__.__name__ + ".run_checkpoint_test() saving checkpoint")
        save_checkpoint(neox_args=args_loaded, iteration=42, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)

        # forward
        context_tokens_tensor = torch.cuda.LongTensor([[1,2,3,4,5],[1,2,3,4,5],[6,7,8,9,10],[1,2,3,4,100]])
        tokens, attention_mask, position_ids = get_batch(args_loaded, context_tokens_tensor)
        output = forward_model(args_loaded, model, (tokens, position_ids, attention_mask))

        # assert outputs are the right shape
        self.assertEqual(output.size(0), 4)
        self.assertEqual(output.size(1), context_tokens_tensor.size(1))
        self.assertTrue(isinstance(output, torch.Tensor))

        # assert correct behaviour
        self.assertTrue(torch.isclose(output[0], output[1]).all().item())
        self.assertFalse(torch.isclose(output[1], output[2]).all().item())
        self.assertTrue(torch.isclose(output[1, 3], output[3, 3]).all().item())
        
        # reload model from checkpoint
        logging.debug(self.__class__.__name__ + ".run_checkpoint_test() reloading checkpoint")
        args_reloaded = NeoXArgs.from_ymls(yaml_list)
        args_reloaded.build_tokenizer()
        args_reloaded.update_value("user_script", str(get_root_directory() / "pretrain_gpt2.py"))
        args_reloaded.update_value("use_cpu_initialization", True)
        args_reloaded.update_value("save", TEST_CHECKPOINT_DIR)
        args_reloaded.update_value("load", TEST_CHECKPOINT_DIR)
        args_reloaded.update_value("log_dir", TEST_LOG_DIR)

        reloaded_model, optimizer, lr_scheduler = setup_model_and_optimizer(neox_args=args_reloaded, inference=False, get_key_value=True)
        iteration = load_checkpoint(neox_args=args_reloaded, model=reloaded_model, optimizer=optimizer, lr_scheduler=lr_scheduler)
        reloaded_model.eval()

        #ensure same checkpoint is loaded
        self.assertEqual(iteration, 42)

        reloaded_output = forward_model(args_reloaded, model, (tokens, position_ids, attention_mask))

        #check re-loaded model returns the same results
        self.assertTrue(torch.isclose(output, reloaded_output).all().item())

        #check all weight groups are the same
        for idx, ((n1, p1), (n2, p2)) in enumerate(zip(list(model.module.named_parameters()), list(reloaded_model.module.named_parameters()))):
            self.assertTrue(n1 == n2)
            params_equal = (p1 == p2).all().item()
            self.assertTrue(params_equal)
            if not params_equal:
                logging.error(self.__class__.__name__ + ".run_checkpoint_test() " + f"layer {idx} {n1} has same different after loading of checkpoint")

    def test_model_small(self):
        self.run_checkpoint_test(get_configs_with_path(["local_setup.yml", "small.yml"]))

    def test_model_medium(self):
        self.run_checkpoint_test(get_configs_with_path(["local_setup.yml", "medium.yml"]))

    def test_model_small(self):
        self.run_checkpoint_test(get_test_configs_with_path(["test_local_setup.yml", "test_small.yml"]))

    def test_model_medium(self):
        self.run_checkpoint_test(get_test_configs_with_path(["test_local_setup.yml", "test_medium.yml"]))

if __name__ == "__main__":
    suite = unittest.TestSuite()

    #Run all required tests
    suite.addTest(TestModelCheckpoint("test_model_small"))
    #suite.addTest(TestModelCheckpoint("test_model_medium"))

    unittest.TextTestRunner(failfast=False).run(suite)