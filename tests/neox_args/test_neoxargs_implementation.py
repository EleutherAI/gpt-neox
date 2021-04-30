import unittest

from megatron.neox_arguments import NeoXArgs


class TestNeoXArgsImplementation(unittest.TestCase):
    """
    verify code implementation of NeoXArgs 
    """
    
    def test_neoxargs_duplicates(self):
        """
        tests that there are no duplicates among parent classes of NeoXArgs
        """
        self.assertTrue(NeoXArgs.validate_keys())