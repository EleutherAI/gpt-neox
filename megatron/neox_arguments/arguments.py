from dataclasses import dataclass
from pathlib import Path

if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.absolute()))

from .deepspeed_runner import NeoXArgsDeepspeedRunnerArguments
#from .deepspeed_config import NeoXArgsDeepspeedConfig
from .model import NeoXArgsModel
from .training import NeoXArgsTraining
from .parallelism import NeoXArgsParallelism
from .other import NeoXArgsOther

@dataclass
class NeoXArgs(
    NeoXArgsDeepspeedRunnerArguments, 
    NeoXArgsModel, 
    NeoXArgsTraining, 
    NeoXArgsParallelism,
    NeoXArgsOther
    ):

    def __init__(self):
        self.calcule_derived()


    def get_deepspeed_args(self):
        pass

    def load_from_yml(self, path_to_yml_file):
        
        self.calcule_derived()

    def calcule_derived(self):
        pass

    @classmethod
    def validate_keys(cls):
        """
        test that there are no duplicate arguments
        """
        source_classes = list(cls.__bases__)
        defined_properties = dict()

        for source_class in source_classes:
            source_vars = list(source_class.__dataclass_fields__)
            for item in source_vars:
                if item in defined_properties.keys():
                    print(f'({cls.__name__}) duplicate of item: {item}, in class {source_class.__name__} and {defined_properties[item]}', flush=True)
                    return False
                else:
                    defined_properties[item] = source_class.__name__
        return True
    
if __name__ == "__main__":

    args = NeoXArgs()
    print("")