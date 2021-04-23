import os
import sys
import unittest

sys.path.append('./megatron/')
from neox_arguments import NeoXArgs

class DuplicateArgumentTest(unittest.TestCase):
    
    def test_duplicates(self):
        self.assertTrue(NeoXArgs.validate_keys())