import os
import sys
import unittest
from unittest.mock import patch

from ..common import get_root_directory, get_config_directory

if __name__ == "__main__":
    sys.path.append(os.path.abspath(''))

from megatron.neox_arguments import NeoXArgs

class TestNeoXArgsCommandLine(unittest.TestCase):

    def test_neoxargs_load_consume_deepy_args(self):
        """
        verify consume_deepy_args processes command line arguments
        """
        with patch('sys.argv', [str(get_root_directory() / "deepy.py"), "pretrain_gpt2.py", '-d', str(get_config_directory()] + ["small.yml", "local_setup.yml"]):
            args_loaded_consume = NeoXArgs.consume_deepy_args()

        args_loaded_yamls = NeoXArgs.from_ymls(yaml_list)
        self.assertRaises(AssertionError, NeoXArgs, (yaml_list, ))

if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(LoadArgumentTest("test_neoxargs_load_consume_deepy_args"))
    unittest.TextTestRunner(failfast=True).run(suite)