from dataclasses import dataclass
from typing import List
from pathlib import Path
import yaml

from .deepspeed_runner import NeoXArgsDeepspeedRunner
from .deepspeed_config import NeoXArgsDeepspeedConfig
from .model import NeoXArgsModel
from .tokenizer import NeoXArgsTokenizer
from .training import NeoXArgsTraining
from .parallelism import NeoXArgsParallelism
from .logging import NeoXArgsLogging
from .other import NeoXArgsOther

@dataclass
class NeoXArgs(
    NeoXArgsDeepspeedRunner, 
    NeoXArgsDeepspeedConfig,
    NeoXArgsModel, 
    NeoXArgsTokenizer,
    NeoXArgsTraining, 
    NeoXArgsParallelism,
    NeoXArgsLogging,
    NeoXArgsOther
    ):
    """
    data class containing all configurations

    NeoXArgs inherits from a number of small configuration classes
    """

    def __post_init__(self):
        self.calcule_derived()

    def get_deepspeed_args(self):
        pass

    @classmethod
    def from_ymls(cls, paths_to_yml_files: List[str]):
        """
        instantiates NeoXArgs while reading values from yml files
        """

        # initialize an empty config dictionary to be filled by yamls
        config = dict()

        # iterate of all to be loaded yaml files
        for conf_file_name in paths_to_yml_files:

            # load file
            with open(conf_file_name) as conf_file:
                conf = yaml.load(conf_file, Loader=yaml.FullLoader)

            # check for key duplicates and load values
            for conf_key, conf_value in conf.items():
                if conf_key in config:
                    raise ValueError(f'Conf file {conf_file_name} has the following duplicate keys with previously loaded file: {conf_key}')

                conf_key_converted = conf_key.replace("-", "_") #TODO remove replace and update configuration files?
                config[conf_key_converted] = conf_value

        # instantiate class and return
        # duplicate values are again checked upon instantiation
        return cls(**config)
        

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
    

    def validate_values(self):
        # the current codebase assumes running with deepspeed only
        if not self.deepspeed:
            return False


        return True
