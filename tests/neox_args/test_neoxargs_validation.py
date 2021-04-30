import unittest

from megatron.neox_arguments import NeoXArgs


class TestNeoXArgsValidation(unittest.TestCase):
    """
    verify the implementation of NeoXArgs
    """

    def test_neoxargs_empty_args_are_not_valid(self):
        """
        NeoXArgs cannot be instantiated without required args
        """
        self.assertRaises(AssertionError, NeoXArgs)