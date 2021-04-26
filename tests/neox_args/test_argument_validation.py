import os
import sys
import unittest

from megatron.neox_arguments import NeoXArgs

class ValidateArgumentTest(unittest.TestCase):

    def test_empty_args_are_not_valid(self):
        """
        NeoXArgs cannot be instantiated without required args
        """
        self.assertRaises(AssertionError, NeoXArgs)