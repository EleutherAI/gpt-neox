import os
import re
import sys
import shutil
import unittest
from unittest.mock import patch
from pathlib import Path

if __name__ == "__main__":
    sys.path.append(os.path.abspath(''))

from megatron.neox_arguments import NeoXArgs
from megatron.global_vars import set_global_variables, get_args, reset_global_variables
from megatron.model import GPT2Model, GPT2ModelPipe
from megatron import initialize_megatron
from megatron import mpu
from megatron.text_generation_utils import get_batch
from megatron.training import setup_model_and_optimizer

from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint
from pretrain_gpt2 import model_provider
from megatron.utils import get_ltor_masks_and_position_ids

from tests.common import get_root_directory, get_configs_with_path
import torch

class TestModelInitialization(unittest.TestCase):

    def test_model_initialization(self):
        reset_global_variables()

        # intitially load config from files as would be the case in deepy.py
        yaml_list = get_configs_with_path(["small.yml", "local_setup.yml"])
        args_loaded = NeoXArgs.from_ymls(yaml_list)
        args_loaded.update_value("user_script", str(get_root_directory() / "pretrain_gpt2.py"))
        args_loaded.update_value("pipe_parallel_size", 0) # overwrite pipeline parameter, config in small.yml may have changed!
        args_loaded.update_value("num_unique_layers", 4)
        args_loaded.update_value("use_cpu_initialization", True)
        #args_loaded.update_value("batch_size", 8)

        args_loaded.update_value("save", "test_checkpoint")
        args_loaded.update_value("load", "test_checkpoint")

        deepspeed_main_args = args_loaded.get_deepspeed_main_args()

        # patch sys.argv so that args can be access by set_global_variables within initialize_megatron
        with patch('sys.argv', deepspeed_main_args):
            initialize_megatron()

        # load args from global variables
        args = get_args()

        # remove any existing checkpoints if they exist
        path = os.path.join(get_root_directory(), args.load)

        shutil.rmtree(path)

        model, optimizer, lr_scheduler = setup_model_and_optimizer(lambda: model_provider(use_wandb=False))
        model.eval()
        
        context_tokens_tensor = torch.cuda.LongTensor([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]])
        tokens, attention_mask, position_ids = get_batch(context_tokens_tensor)
        output = model(tokens, position_ids, attention_mask)

        # assert outputs are the right shape
        self.assertEqual(output.size(0), args.batch_size)
        self.assertEqual(output.size(1), context_tokens_tensor.size(1))
        # assert model output is a tensor
        self.assertTrue(isinstance(output, torch.Tensor))

        for n, p in model.module.named_parameters():
            if n == "final_linear.weight":
                print(n, p, flush=True)

        print(model.module.state_dict_for_save_checkpoint()["language_model"]["transformer"]["final_linear.weight"])
        print(model.module.state_dict()["final_linear.weight"])
        
        # save model checkpoint
        save_checkpoint(42, model, optimizer, lr_scheduler)
        
        # reload model and test forward pass
        reloaded_model, optimizer, lr_scheduler = setup_model_and_optimizer(lambda: model_provider(use_wandb=False))
        iteration = load_checkpoint(reloaded_model, optimizer, lr_scheduler)

        #ensure same checkpoint is loaded
        self.assertEqual(iteration, 42)

        reloaded_model.eval()

        reloaded_output = reloaded_model(tokens, position_ids, attention_mask)

        #print(output)
        #print(reloaded_output)  
        #print(torch.isclose(output, reloaded_output))    

        for idx, ((n1, p1), (n2, p2)) in enumerate(zip(list(model.module.named_parameters()), list(reloaded_model.module.named_parameters()))):
            # It's the 'final linear layer' of the gpt2model, although not sure how to iterate through param group names
            self.assertTrue(n1 == n2)
            params_equal = (p1 == p2).all().item()
            if not params_equal:
                print("")
            self.assertTrue(params_equal, f"test_model_checkpoint() layer {idx} {n1} has same parameters after loading of checkpoint")

        self.assertTrue(torch.isclose(output, reloaded_output).all().item())

        #TODO test changing batch size, because i had some weird experience with this last time
        


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestModelInitialization("test_model_initialization"))
    unittest.TextTestRunner(failfast=True).run(suite)