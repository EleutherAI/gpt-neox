import os
import yaml
import json
import logging
import shortuuid

from dataclasses import dataclass
from typing import List
from pathlib import Path
from socket import gethostname

import torch

from deepspeed.launcher.runner import DLTS_HOSTFILE

from megatron.logging import Tee
from megatron.utils import obtain_resource_pool

from .deepspeed_runner import NeoXArgsDeepspeedRunner
from .deepspeed_config import NeoXArgsDeepspeedConfig
from .model import NeoXArgsModel
from .tokenizer import NeoXArgsTokenizer
from .training import NeoXArgsTraining
from .parallelism import NeoXArgsParallelism
from .logging import NeoXArgsLogging
from .other import NeoXArgsOther


ZERO_DEFAULTS = {
    "stage": 0,
    "allgather_partitions": True,
    "reduce_scatter": True,
    "allgather_bucket_size": int(5e8),
    "overlap_comm": False,
    "reduce_scatter": True,
    "reduce_bucket_size": int(5e8),
    "contiguous_gradients": False,
    "cpu_offload": False
}

GRADIENT_CLIPPING_DEFAULT = 1.0 #TODO this is different from the default value for deepspeed

OPT_DEFAULT = "adam"
OPT_PARAMS_DEFAULTS = {
    "lr": 0.001,
    "betas": [
        0.9,
        0.999
    ],
    "eps": 1e-8,
    "weight_decay": 0,
    "freeze_step": 400,
    "momentum": 0.0,
    "cuda_aware": False
}
OPTIMIZER_OPTIONS = ["adam", "onebitadam", "cpu_adam", "cpu_torch_adam"]


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
        """
        after initialization of default or loaded values 
        a number of function are performed in order to 
        calculate values and  assert consistency
        """
        if not NeoXArgs.validate_keys():
            raise ValueError(self.__class__.__name__+".__post_init__() NeoXArgs keys cannot be validated")

        self.enable_logging()
        self.configure_distributed_args()
        self.calculated_derived()
        self.validate_values()
        self.save_yml()

    @property
    def deepspeed_config(self) -> str:
        return "test"

    def get_deepspeed_args(self):
        pass

    def update_value(self, key: str, value):
        """
        updates a property value if the key is already existing

        Problem: a previously non-existing property can be added to the class instance without error. 
        """
        if hasattr(self, key):
            # TODO make sure that only Nones are overwritten or warn
            setattr(self, key, value)
        else:
            error_message = self.__class__.__name__+".update_value() to be updated property "+str(key)+" does not exist"
            logging.error(error_message)
            raise ValueError(error_message)

    @classmethod
    def from_ymls(cls, paths_to_yml_files: List[str]):
        """
        instantiates NeoXArgs while reading values from yml files
        """

        print(cls.__name__+".from_ymls() "+str(paths_to_yml_files), flush=True)

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

        #TODO check for unspecified params?

        # instantiate class and return
        # duplicate values and unrecognized keys are again checked upon instantiation
        return cls(**config)
        
    def enable_logging(self):
        """
        enable Tee logs based on the configured logdir
        """
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            hostname = gethostname()
            file_prefix = os.path.join(self.log_dir, hostname)
            Tee(file_prefix+'_stdout.txt', err=False)
            Tee(file_prefix + '_stderr.txt', err=True)

    def save_yml(self):
        """
        saves the configured value to the configured save directory (if any)
        """
        if self.save is not None:
            os.makedirs(self.save, exist_ok=True)
            config_file = os.path.join(self.save, 'config.yml')
            with open(config_file, 'w') as f:
                json.dump(vars(self), f, indent=4)

    def configure_distributed_args(self):
        if self.deepspeed_mpi:
            from deepspeed.utils.distributed import mpi_discovery
            mpi_discovery()
        
        self.update_value("local_rank", int(os.getenv('LOCAL_RANK', '0')))
        self.update_value("rank", int(os.getenv('RANK', '0')))
        self.update_value("world_size", int(os.getenv("WORLD_SIZE", '1')))
        self.update_value("model_parallel_size", min(self.model_parallel_size, self.world_size))

        if self.rank == 0:
            print(self.__class__.__name__+".configure_distributed_args() using world size: {} and model-parallel size: {} ".format(self.world_size, self.model_parallel_size), flush=True)

    @staticmethod
    def calculate_batch_parameters(world_size, train_batch=None, micro_batch=None, grad_acc=None):
        # all values are provided nothing needs to be set
        if train_batch is not None and \
                micro_batch is not None and \
                grad_acc is not None:
            return train_batch, micro_batch, grad_acc

        # gradient_accumulation_steps needs to be set
        elif train_batch is not None and \
                micro_batch is not None:
            grad_acc = train_batch // micro_batch
            grad_acc //= world_size

        # micro_batch_per_gpu needs to be set
        elif train_batch is not None and \
                grad_acc is not None:
            micro_batch = train_batch // world_size
            micro_batch //= grad_acc

        # train_batch_size needs to be set
        elif micro_batch is not None and \
                grad_acc is not None:
            train_batch = micro_batch * grad_acc
            train_batch *= world_size

        # gradient_accumulation_steps and micro_batch_per_gpus is set
        elif train_batch is not None:
            grad_acc = 1
            micro_batch = train_batch // world_size

        # train_batch_size and gradient_accumulation_step is set
        elif micro_batch is not None:
            train_batch = micro_batch * world_size
            grad_acc = 1

        # either none of the three parameters are provided or just gradient_accumulation_step is provided
        else:
            assert False, \
                'Either train_batch_size or micro_batch_per_gpu needs to be provided'
        return train_batch, micro_batch, grad_acc

    @staticmethod
    def check_batch_parameters(world_size, train_batch, micro_batch, grad_acc):
        assert train_batch > 0, \
            f'Train batch size: {train_batch} has to be greater than 0'

        assert micro_batch > 0, \
            f'Micro batch size per gpu: {micro_batch} has to be greater than 0'

        assert grad_acc > 0, \
            f'Gradient accumulation steps: {grad_acc} has to be greater than 0'

        assert train_batch == micro_batch * grad_acc * world_size, \
            (f'Check batch related parameters. train_batch_size is not equal'
            ' to micro_batch_per_gpu * gradient_acc_step * world_size'
            f'{train_batch} != {micro_batch} * {grad_acc} * {world_size}')

    def calculated_derived(self):
        """
        calculates configuration values depending on so far existing configuration
        """

        # wandb
        # sets a unique wanddb group
        if self.wandb_group is None:
            # if none is defined a uuid is set for the run
            self.wandb_group = shortuuid.uuid()
        else:
            # if one is defined it is concatenated with a uuid to make the run unique
            self.wandb_group = str(self.wandb_group) + shortuuid.uuid()

        # number of gpus
        # Get number of GPUs param or hostfile to determine train_batch_size
        num_gpus = self.num_gpus
        if num_gpus is None:
            num_gpus = -1 # set -1 for backwards compatibility to old default value
        if num_gpus < 1:
            if self.hostfile is not None or os.path.exists(DLTS_HOSTFILE):
                hostfile_path = self.hostfile or DLTS_HOSTFILE
                resources = obtain_resource_pool(hostfile_path, self.include or "", self.exclude or "")
                num_gpus = sum(map(len, resources.values()))
            else:
                num_gpus = torch.cuda.device_count()
        self.update_value("num_gpus", num_gpus)

        logging.info(self.__class__.__name__+".calcule_derived() "+f"Total number of GPUs determined to be: {self.num_gpus}")

        # get world size in the model/pipe parallel case, the actual `world size` deepspeed uses is the size of the
        # data-parallel group, or (num_gpus / mp_size) / pp_size
        pp_size = self.pipe_parallel_size
        pp_size = pp_size if pp_size >= 1 else 1
        mp_size = self.model_parallel_size
        mp_size = mp_size if mp_size >= 1 else 1
                      
        # pp_size and mp_size are only used here to compute world_size and nowhere else. The way that these values actually get to deepspeed
        # is through convert_to_old_args. The entire chain of how that happens:
        # https://github.com/EleutherAI/gpt-neox/blob/2ceefba0ef12b94eb35a518f7dea9f34fc43c9af/megatron/arguments.py#L430
        # https://github.com/EleutherAI/gpt-neox/blob/2ceefba0ef12b94eb35a518f7dea9f34fc43c9af/megatron/arguments.py#L45
        # https://github.com/EleutherAI/gpt-neox/blob/2ceefba0ef12b94eb35a518f7dea9f34fc43c9af/megatron/config_monster.py#L17
        # https://github.com/EleutherAI/gpt-neox/blob/2ceefba0ef12b94eb35a518f7dea9f34fc43c9af/megatron/config_monster.py#L40
        # https://github.com/EleutherAI/gpt-neox/blob/2ceefba0ef12b94eb35a518f7dea9f34fc43c9af/megatron/config_monster.py#L330

        world_size = ((num_gpus / pp_size) / mp_size)
        if not (world_size % 1 == 0):
            error_message = self.__class__.__name__+".calcule_derived() "+f"(num_gpus / pp_size) / mp_size [({num_gpus} / {pp_size}) / {mp_size}] must be a whole number"
            logging.error(error_message)
            raise AssertionError(error_message)
        self.update_value("world_size", int(world_size))

        # Automatically derive train_batch_size = train_micro_batch_size_per_gpu*num_gpus*gradient_accumulation_steps
        train_batch_size, train_micro_batch_size_per_gpu, gradient_accumulation_steps = self.calculate_batch_parameters(
            world_size=world_size, 
            train_batch=self.train_batch_size, 
            micro_batch=self.train_micro_batch_size_per_gpu, 
            grad_acc=self.gradient_accumulation_steps
            )
        self.check_batch_parameters(
            world_size=world_size, 
            train_batch=train_batch_size, 
            micro_batch=train_micro_batch_size_per_gpu, 
            grad_acc=gradient_accumulation_steps
        )
        self.update_value("train_batch_size", train_batch_size)
        self.update_value("train_micro_batch_size_per_gpu", train_micro_batch_size_per_gpu)
        self.update_value("gradient_accumulation_steps", gradient_accumulation_steps)

        self.update_value("batch_size", train_micro_batch_size_per_gpu)
      
        # duplicated items
        self.update_value("half_precision", (self.fp16 or {}).get("enabled", False))
        self.update_value("gas", self.gradient_accumulation_steps)
        self.update_value("clip_grad", self.gradient_clipping or GRADIENT_CLIPPING_DEFAULT)

        # zero optimization
        if self.zero_optimization is None:
            self.zero_optimization = ZERO_DEFAULTS # a dict is overwritten and not updated key by key
        #TODO this could lead to an inconsistency as only part of zero optimization could be defined in input config
        self.update_value("zero_stage", self.zero_optimization.get('stage', ZERO_DEFAULTS['stage']))
        self.update_value("zero_reduce_scatter", self.zero_optimization.get('reduce_scatter', ZERO_DEFAULTS['reduce_scatter']))
        self.update_value("zero_contiguous_gradients", self.zero_optimization.get('contiguous_gradients', ZERO_DEFAULTS['contiguous_gradients']))
        self.update_value("zero_reduce_bucket_size", self.zero_optimization.get('reduce_bucket_size', ZERO_DEFAULTS['reduce_bucket_size']))
        self.update_value("zero_allgather_bucket_size", self.zero_optimization.get('allgather_bucket_size', ZERO_DEFAULTS['allgather_bucket_size']))

        # optimizer and scheduler
        opt_params = self.optimizer or {"type": OPT_DEFAULT, "params": OPT_PARAMS_DEFAULTS}
        #TODO why is self.optimizer not updated if not defined?
        #TODO this could lead to an inconsistency as only part of optimizater could be defined in input config
        self.update_value("lr", opt_params['params'].get('lr', OPT_PARAMS_DEFAULTS['lr']))
        self.update_value("adam_beta1", opt_params['params'].get('betas', OPT_PARAMS_DEFAULTS['betas'])[0])
        self.update_value("adam_beta2", opt_params['params'].get('betas', OPT_PARAMS_DEFAULTS['betas'])[1])
        self.update_value("adam_eps", opt_params['params'].get('eps', OPT_PARAMS_DEFAULTS['eps']))
        self.update_value("momentum", opt_params['params'].get('momentum', OPT_PARAMS_DEFAULTS['momentum']))
        
        self.update_value("onebitadam", opt_params["type"].lower() == "onebitadam")
        self.update_value("cpu_optimizer", opt_params["type"].lower() == "cpu_adam")
        self.update_value("cpu_torch_adam", opt_params["type"].lower() == "cpu_torch_adam")
        self.update_value("sm3", opt_params["type"].lower() == "sm3")
            
        if opt_params["type"].lower() == "onebitadam":
            # onebitadam needs to instantiated by deepspeed, and so we need to pass deepspeed scheduler args
            # for all other optimizers, the scheduling is handled by megatron
            self.scheduler = {
                "type": "WarmupDecayLR",  # for now this is the only ds scheduler offering decay
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": opt_params["params"]["lr"],
                    "warmup_num_steps": int(self.train_iters * self.warmup),
                    "total_num_steps": self.lr_decay_iters or self.train_iters
            }}

        # Fp16 loss scaling.
        #if self.loss_scale is None:
        #    self.update_value("dynamic_loss_scale", True)
        #else:
        #    self.update_value("dynamic_loss_scale", False)

        print("")

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
                    error_message = f'({cls.__name__}) duplicate of item: {item}, in class {source_class.__name__} and {defined_properties[item]}'
                    logging.error(error_message)
                    return False
                else:
                    defined_properties[item] = source_class.__name__
        return True
    
    def validate_values(self):
        # the current codebase assumes running with deepspeed only
        if not self.deepspeed:
            return False

        if self.lr is None:
            error_message = self.__class__.__name__+".validate_values() lr is None"
            logging.error(error_message)
            raise ValueError(error_message)
            return False

        if (self.optimizer or {}).get("type", "").lower() not in OPTIMIZER_OPTIONS:
            error_message = self.__class__.__name__+".validate_values() "+f'Optimizer type {opt_params["type"]} not recognized, please choose from: \n {OPTIMIZER_OPTIONS}'
            logging.error(error_message)
            raise ValueError(error_message)
            return False


        return True
