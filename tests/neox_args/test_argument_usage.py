import os
import re
import sys
import unittest

sys.path.append('./megatron/')
from neox_arguments import NeoXArgs

class ArgumentUsageTest(unittest.TestCase):
    
    def test_usage(self):
        
        # collect files
        files = [] 
        foldersToCheck = ['./megatron/'] 
        while (len(foldersToCheck) > 0): 
            for (dirpath, dirnames, filenames) in os.walk(foldersToCheck[0]): 
                while(len(dirnames) > 0): 
                    foldersToCheck.append(foldersToCheck[0] + dirnames[0] + "/") 
                    del dirnames[0] 
                while(len(filenames) > 0): 
                    if filenames[0].endswith('py'):
                        files.append(foldersToCheck[0] + filenames[0]) 
                    
                    del filenames[0] 
                del foldersToCheck[0] 

        # remove files from test
        files.remove('./megatron/text_generation_utils.py')
        files.remove('./megatron/tokenizer/train_tokenizer.py')

        declared_all = True
        exclude = ['params_dtype', 'deepspeed_config', 'get', 'pop']

        for f in files:
            out = self.run_test(f)

            for item in exclude:
                if item in out:
                    out.remove(item)

            if out != []:
                print(f"(arguments used not in neox args): {f}: {out}", flush=True)
                declared_all = False

        self.assertTrue(declared_all)

    def check_file(self, file):
        with open(file, 'r') as f:
            text = f.read()
        matches = re.findall(r"(?<=args\.).{2,}?(?=[\s\n(){}+-/*;:,=])", text)
        return list(dict.fromkeys(matches))

    def run_test(self, file):
        neox_args = list(NeoXArgs.__dataclass_fields__)
        missing = []
        matches = self.check_file(file)
        for match in matches:
            if match not in neox_args:
                missing.append(match)
        return missing