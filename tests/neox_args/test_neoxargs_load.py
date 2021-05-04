import os
import sys
import unittest

if __name__ == "__main__":
    sys.path.append(os.path.abspath(''))

from megatron.neox_arguments import NeoXArgs


from tests.common import get_configs_with_path

class TestNeoXArgsLoad(unittest.TestCase):
    """
    verify loading of yaml files
    """

    def test_neoxargs_load_arguments_small_local_setup(self):
        """
        verify small.yml can be loaded without raising validation errors
        """
        yaml_list = get_configs_with_path(["small.yml", "local_setup.yml"])
        args_loaded = NeoXArgs.from_ymls(yaml_list)
        self.assertTrue(isinstance(args_loaded, NeoXArgs))

    def test_neoxargs_load_arguments_small_local_setup_text_generation(self):
        """
        verify small.yml can be loaded together with text generation without raising validation errors
        """
        yaml_list = get_configs_with_path(["small.yml", "local_setup.yml", "text_generation.yml"])
        args_loaded = NeoXArgs.from_ymls(yaml_list)
        self.assertTrue(isinstance(args_loaded, NeoXArgs))
        
    def test_neoxargs_load_arguments_medium_local_setup(self):
        """
        verify medium.yml can be loaded without raising validation errors
        """
        yaml_list = get_configs_with_path(["medium.yml", "local_setup.yml"])
        args_loaded = NeoXArgs.from_ymls(yaml_list)
        self.assertTrue(isinstance(args_loaded, NeoXArgs))

    def test_neoxargs_load_arguments_large_local_setup(self):
        """
        verify large.yml can be loaded without raising validation errors
        """
        yaml_list = get_configs_with_path(["large.yml", "local_setup.yml"])
        args_loaded = NeoXArgs.from_ymls(yaml_list)
        self.assertTrue(isinstance(args_loaded, NeoXArgs))

    def test_neoxargs_load_arguments_2_7B_local_setup(self):
        """
        verify 2-7B.yml can be loaded without raising validation errors
        """
        yaml_list = get_configs_with_path(["2-7B.yml", "local_setup.yml"])
        args_loaded = NeoXArgs.from_ymls(yaml_list)
        self.assertTrue(isinstance(args_loaded, NeoXArgs))

    def test_neoxargs_load_arguments_6_7B_local_setup(self):
        """
        verify 6-7B.yml can be loaded without raising validation errors
        """
        yaml_list = get_configs_with_path(["6-7B.yml", "local_setup.yml"])
        args_loaded = NeoXArgs.from_ymls(yaml_list)
        self.assertTrue(isinstance(args_loaded, NeoXArgs))

    def test_neoxargs_load_arguments_13B_local_setup(self):
        """
        verify 13B.yml can be loaded without raising validation errors
        """
        yaml_list = get_configs_with_path(["13B.yml", "local_setup.yml"])
        args_loaded = NeoXArgs.from_ymls(yaml_list)
        self.assertTrue(isinstance(args_loaded, NeoXArgs))

    def test_neoxargs_load_arguments_XL_local_setup(self):
        """
        verify XL.yml can be loaded without raising validation errors
        """
        yaml_list = get_configs_with_path(["XL.yml", "local_setup.yml"])
        args_loaded = NeoXArgs.from_ymls(yaml_list)
        self.assertTrue(isinstance(args_loaded, NeoXArgs))

    def test_neoxargs_load_arguments_175B_local_setup(self):
        """
        verify 13B.yml can be loaded without raising validation errors
        """
        yaml_list = get_configs_with_path(["175B.yml", "local_setup.yml"])
        args_loaded = NeoXArgs.from_ymls(yaml_list)
        self.assertTrue(isinstance(args_loaded, NeoXArgs))

    def test_neoxargs_load_local_setup_only(self):
        """
        verify assertion error if required arguments are not provided
        """
        yaml_list = get_configs_with_path(["local_setup.yml"])
        self.assertRaises(AssertionError, NeoXArgs, (yaml_list, ))
    
if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(LoadArgumentTest("test_neoxargs_load_consume_deepy_args"))
    unittest.TextTestRunner(failfast=True).run(suite)