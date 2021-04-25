import os
import sys
import unittest

sys.path.append('./megatron/')
from neox_arguments import NeoXArgs

class ValidateArgumentTest(unittest.TestCase):
    #TODO implement test configs to read from
    def test_empty_args_are_not_valid(self):
        self.assertTrue(NeoXArgs())