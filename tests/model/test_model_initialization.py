import os
import re
import sys
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
  #args_loaded.update_value("fp16", False)
        deepspeed_main_args = args_loaded.get_deepspeed_main_args()

        # patch sys.argv so that args can be access by set_global_variables within initialize_megatron
        with patch('sys.argv', deepspeed_main_args):
            initialize_megatron()

        # load args from global variables
        args = get_args()

        self.assertTrue(isinstance(args, NeoXArgs))

        model = GPT2Model(num_tokentypes=0, parallel_output=True, inference=False, get_key_value=True)
        
        self.assertTrue(isinstance(model, GPT2Model)) 

        #TODO check model setup
        self.assertEqual(model.language_model.transformer.num_layers, args.num_layers)
        self.assertIsNotNone(model.language_model.transformer.num_unique_layers)
        self.assertEqual(len(model.language_model.transformer.layers), args.num_unique_layers)
        self.assertEqual(model.language_model.transformer.checkpoint_activations, args.checkpoint_activations)
        self.assertEqual(model.language_model.hidden_size, args.hidden_size)
        self.assertEqual(model.language_model.hidden_size, args.hidden_size)
        self.assertEqual(model.language_model.embedding_type, args.pos_emb)
        self.assertNotEqual(model.weight_tying, args.no_weight_tying)
        self.assertEqual(model.fp16_lm_cross_entropy, args.fp16_lm_cross_entropy)

        #TODO run one forward pass
        # Not currently functioning
        
        #model = model.cuda()
        #input_ids = torch.zeros((1,100), dtype=torch.long).cuda()
        #tokens, attention_mask, position_ids = get_batch(input_ids)

        #position_ids = torch.zeros((1,100), dtype=torch.long).cuda()
        #attention_mask = torch.ones((1,100), dtype=torch.long).cuda()
        #print(model.forward(tokens, attention_mask, position_ids))

        


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestModelInitialization("test_model_initialization"))
    unittest.TextTestRunner(failfast=True).run(suite)