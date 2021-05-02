import os
import re
import sys
import unittest

if __name__ == "__main__":
    sys.path.append(os.path.abspath(''))

from megatron.neox_arguments import NeoXArgs
from tests.common import get_root_directory

class TestNeoXArgsArgumentUsage(unittest.TestCase):
    """
    plausibility checks on the usage of arguments within code
    """
    
    def test_neoxargs_usage(self):
        """"
        checks for code pieces of the pattern "args.*" and verifies that such used arg is defined in NeoXArgs
        """

        
        declared_all = True
        neox_args_attributes = set(NeoXArgs.__dataclass_fields__.keys())

        # we exlude a number of properties (implemented with the @property decorator) or functions that we know exists
        exclude = set(['params_dtype', 'deepspeed_config', 'get', 'pop', 'get_deepspeed_main_args', 'optimizer["params"]', 'attention_config[layer_number]', 'adlr_autoresume_object', 'update_value', 'all_config', 'tensorboard_writer', 'tokenizer'])

        # test file by file
        for filename in (get_root_directory() / "megatron").glob('**/*.py'):
            if filename.name in ["text_generation_utils.py", "train_tokenizer.py"]: continue

            # load file
            with open(filename, 'r') as f:
                file_contents = f.read()

            # find args matches
            matches = list(re.findall(r"(?<=args\.).{2,}?(?=[\s\n(){}+-/*;:,=])", file_contents))
            if len(matches) == 0: continue

            # compare
            for match in matches:
                if match not in neox_args_attributes and match not in exclude:
                    print(f"(arguments used not found in neox args): {filename.name}: {match}", flush=True)
                    declared_all = False

        self.assertTrue(declared_all)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestNeoXArgsArgumentUsage("test_neoxargs_usage"))
    unittest.TextTestRunner(failfast=True).run(suite)