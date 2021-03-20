# Copyright (c) 2021, EleutherAI contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import logging
import yaml
from deepspeed.launcher.runner import DLTS_HOSTFILE

from megatron.utils import obtain_resource_pool
from megatron.arguments import _get_parser
import torch

log = logging.getLogger('ConfigMonster')


def _get_megatron_keys(_megatron_keys_exclude):
    _megatron_keys = list(_get_parser()._option_string_actions.keys())
    _megatron_keys = [item.replace('--', '') for item in _megatron_keys]
    for item in _megatron_keys_exclude:
        try:
            _megatron_keys.remove(item)
        except ValueError:
            pass
    return _megatron_keys


ds_runner_keys = ['hostfile', 'include', 'exclude', 'num_nodes', 'num_gpus', 'master_port', 'master_addr', 'launcher',
                  'launcher_args', 'detect_nvlink_pairs']  # handle separately: 'user_script', 'user_args'

megatron_keys_exclude = [
    'fp16',  # Duplicated in ds_config
    'gas',  # Duplicate of `gradient_accumulation_steps` in ds_config,
    '-h', 'help'  # Argparse arguments - unneeded
          'zero-stage', 'zero-reduce-scatter', 'zero-contiguous-gradients',
    'zero-reduce-bucket-size', 'zero-allgather-bucket-size',  # all zero params from ds_config
    'clip-grad',
    'deepspeed'
]

megatron_keys = _get_megatron_keys(megatron_keys_exclude)

# DS Config manually taken from https://www.deepspeed.ai/docs/config-json/ plus some undocumented keys
ds_config_keys = ['train_batch_size', 'train_micro_batch_size_per_gpu', 'gradient_accumulation_steps', 'optimizer',
                  'scheduler', 'fp32_allreduce', 'prescale_gradients', 'gradient_predivide_factor', 'sparse_gradients',
                  'fp16', 'amp', 'gradient_clipping', 'zero_optimization', 'steps_per_print', 'wall_clock_breakdown',
                  'dump_state', 'flops_profiler', 'activation_checkpointing', 'sparse_attention',
                  'zero_allow_untested_optimizer', ]

neox_config_keys = ['wandb_group', 'wandb_team', 'git_hash']

ds_runner_keys_exclude = []

ds_config_keys_exclude = []

#############
# DEFAULTS: #
#############

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

GRADIENT_CLIPPING_DEFAULT = 1.0

OPTIMIZER_OPTIONS = ["adam", "onebitadam", "cpu_adam", "cpu_torch_adam"]

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


def _set_zero_params(ds_conf, megatron_conf):
    """
    sets the zero params in the megatron args from deepspeed-style params
    """
    ds_zero_params = ds_conf.get("zero_optimization", ZERO_DEFAULTS)
    megatron_conf['zero-stage'] = ds_zero_params.get('stage', ZERO_DEFAULTS['stage'])
    megatron_conf['zero-reduce-scatter'] = ds_zero_params.get('reduce_scatter', ZERO_DEFAULTS['reduce_scatter'])
    megatron_conf['zero-contiguous-gradients'] = ds_zero_params.get('contiguous_gradients',
                                                                    ZERO_DEFAULTS['contiguous_gradients'])
    megatron_conf['zero-reduce-bucket-size'] = ds_zero_params.get('reduce_bucket_size',
                                                                  ZERO_DEFAULTS['reduce_bucket_size'])
    megatron_conf['zero-allgather-bucket-size'] = ds_zero_params.get('allgather_bucket_size',
                                                                     ZERO_DEFAULTS['allgather_bucket_size'])


def _megatron_to_ds_scheduler_args(ds_conf, megatron_conf):
    """
    in the case of onebitadam, deepspeed needs to initialize the lr scheduler (because reasons).
    This function converts the megatron style scheduler args to deepspeed ones.
    """
    opt_params = ds_conf.get("optimizer", {"type": OPT_DEFAULT, "params": OPT_PARAMS_DEFAULTS})
    ds_conf["scheduler"] = {
        "type": "WarmupDecayLR",  # for now this is the only ds scheduler offering decay
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": opt_params["params"]["lr"],
            "warmup_num_steps": int(megatron_conf["train-iters"] * megatron_conf["warmup"]),
            "total_num_steps": megatron_conf.get("lr-decay-iters", megatron_conf["train-iters"])
        }}


def _set_scheduler_params(ds_conf, megatron_conf):
    """
    Sets deepspeed style lr scheduler params from megatron style params
    """
    opt_params = ds_conf.get("optimizer", {"type": OPT_DEFAULT, "params": OPT_PARAMS_DEFAULTS})
    if opt_params["type"].lower() == "onebitadam":
        # onebitadam needs to instantiated by deepspeed, and so we need to pass deepspeed scheduler args
        # for all other optimizers, the scheduling is handled by megatron
        _megatron_to_ds_scheduler_args(ds_conf, megatron_conf)


def _set_optimizer_params(ds_conf, megatron_conf):
    """
    Sets megatron style optimizer params from deepspeed style params
    """
    opt_params = ds_conf.get("optimizer", {"type": OPT_DEFAULT, "params": OPT_PARAMS_DEFAULTS})
    megatron_conf['lr'] = opt_params['params'].get('lr', OPT_PARAMS_DEFAULTS['lr'])
    megatron_conf['adam-beta1'] = opt_params['params'].get('betas', OPT_PARAMS_DEFAULTS['betas'])[0]
    megatron_conf['adam-beta2'] = opt_params['params'].get('betas', OPT_PARAMS_DEFAULTS['betas'])[1]
    megatron_conf['adam-eps'] = opt_params['params'].get('eps', OPT_PARAMS_DEFAULTS['eps'])
    megatron_conf['momentum'] = opt_params['params'].get('momentum', OPT_PARAMS_DEFAULTS['momentum'])

    assert megatron_conf['lr'] is not None
    if opt_params["type"].lower() == "adam":
        pass
    elif opt_params["type"].lower() == "onebitadam":
        megatron_conf['onebitadam'] = True
    elif opt_params["type"].lower() == "cpu_adam":
        megatron_conf['cpu-optimizer'] = True
    elif opt_params["type"].lower() == "cpu_torch_adam":
        megatron_conf['cpu_torch_adam'] = True
    elif opt_params["type"].lower() == "sm3":
        megatron_conf['sm3'] = True
    else:
        raise ValueError(
            f'Optimizer type {opt_params["type"]} not recognized, please choose from: \n {OPTIMIZER_OPTIONS}')


def _batch_assertion(world_size, train_batch, micro_batch, grad_acc):
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


def _set_batch_parameters(world_size, train_batch=None, micro_batch=None, grad_acc=None):
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


def _configure_train_batch_size(world_size, train_batch=None, micro_batch=None, grad_acc=None):
    """
    Configures batch size related parameters and checks for correctness.
    Modified from deepspeed.DeepSpeedConfig._set_batch_related_parameters.
    """
    train_batch, micro_batch, grad_acc = _set_batch_parameters(world_size, train_batch, micro_batch, grad_acc)
    print(train_batch, micro_batch, grad_acc)
    _batch_assertion(world_size, train_batch=train_batch, micro_batch=micro_batch, grad_acc=grad_acc)
    return train_batch, micro_batch, grad_acc


class ConfigMonster:
    """ Clearing up megatron's config monstrosity. """

    def __init__(self):
        pass

    @staticmethod
    def construct_arg_parser():
        parser = argparse.ArgumentParser(description='GPT-NEOX Configuration',
                                         allow_abbrev=False)

        parser.add_argument("user_script",
                            type=str,
                            help="User script to launch, followed by any required "
                                 "arguments.")

        parser.add_argument("--conf_dir", '-d',
                            type=str,
                            default=None,
                            help="Directory to prefix to all configuration file paths")

        parser.add_argument("conf_file",
                            type=str,
                            nargs='+',
                            help="Configuration file path. Multiple files can be provided and will be merged.")

        return parser

    @staticmethod
    def parse_args(parser: argparse.ArgumentParser, args=None, extra_conf=None):
        """
        Parse User Arguments
        """
        args = parser.parse_args(args)

        # Validate user_script exists
        assert os.path.exists(args.user_script), f"User script could not be found: {args.user_script}"

        conf_files = args.conf_file
        if args.conf_dir:
            conf_files = [os.path.join(args.conf_dir, f) for f in conf_files]

        # enables us to pass in `small` instead of `small.yml`
        for cf in conf_files:
            if not cf.endswith('.yml'):
                cf += '.yml'

        # Load and merge all configuration
        conf = {} if extra_conf is None else extra_conf
        for path in conf_files:
            with open(path) as f:
                conf_i = yaml.load(f, Loader=yaml.FullLoader)

            # Check there is are no duplicate keys
            confs_keys = set(conf.keys())
            conf_i_keys = set(conf_i.keys())
            key_intersection = confs_keys.intersection(conf_i_keys)
            assert len(key_intersection) == 0, f'Conf file {path} has duplicate keys with previously ' \
                                               f'loaded file:  {key_intersection}'

            conf.update(conf_i)

        # Assert there are no keys that are not recognised
        unrecognised_keys = [key for key in conf.keys()
                             if key not in ds_runner_keys + megatron_keys + ds_config_keys + neox_config_keys]
        assert len(unrecognised_keys) == 0, f"Configuration parameters not recognised: {', '.join(unrecognised_keys)}"

        # Configuration parameters not specified
        params_missing = [key for key in ds_runner_keys + megatron_keys + ds_config_keys + neox_config_keys
                          if key not in conf]
        if len(params_missing) > 0:
            log.debug(f'Configuration parameters not specified: {", ".join(params_missing)}')

        return args, conf

    @staticmethod
    def derive_params_and_split(conf):
        """
        Derive and insert implicit parameters
        """

        # Get number of GPUs param or hostfile to determine train_batch_size
        num_gpus = conf.get('num_gpus')
        if num_gpus is None and ('hostfile' in conf or os.path.exists(DLTS_HOSTFILE)):
            hostfile_path = conf.get('hostfile', DLTS_HOSTFILE)
            resources = obtain_resource_pool(hostfile_path, conf.get('include', ''), conf.get('exclude', ''))
            num_gpus = sum(map(len, resources.values()))
        else:
            num_gpus = torch.cuda.device_count()
            conf["num_gpus"] = num_gpus

        log.info(f"Total number of GPUs determined to be: {num_gpus}")

        # get world size in the model/pipe parallel case, the actual `world size` deepspeed uses is the size of the
        # data-parallel group, or (num_gpus / mp_size) / pp_size
        pp_size = conf.get('pipe-parallel-size', 0)
        pp_size = pp_size if pp_size >= 1 else 1
        mp_size = conf.get('model-parallel-size', 0)
        mp_size = mp_size if mp_size >= 1 else 1
        world_size = ((num_gpus / pp_size) / mp_size)
        assert world_size % 1 == 0, f"(num_gpus / pp_size) / mp_size [({num_gpus} / {pp_size}) / {mp_size}] must be a whole number"

        # Automatically derive train_batch_size = train_micro_batch_size_per_gpu*num_gpus*gradient_accumulation_steps
        conf['train_batch_size'], conf['train_micro_batch_size_per_gpu'], conf[
            'gradient_accumulation_steps'] = _configure_train_batch_size(world_size, conf.get('train_batch_size'),
                                                                         conf.get('train_micro_batch_size_per_gpu'),
                                                                         conf.get('gradient_accumulation_steps'))
        conf['gradient_accumulation_steps'] = int(conf['gradient_accumulation_steps'])
        conf['batch-size'] = conf['train_micro_batch_size_per_gpu']  # we need to pass batch size into megatron

        ds_runner_conf = {key: conf[key] for key in ds_runner_keys if key in conf}
        megatron_conf = {key: conf[key] for key in megatron_keys + neox_config_keys if key in conf}
        ds_config_conf = {key: conf[key] for key in ds_config_keys if key in conf}

        # Items duplicated
        megatron_conf['deepspeed'] = True # should always be using deepspeed
        ds_config_conf['deepspeed'] = True
        megatron_conf['fp16'] = conf.get('fp16', {}).get('enabled', False)
        megatron_conf['gas'] = conf.get('gradient_accumulation_steps')
        _set_zero_params(ds_config_conf, megatron_conf)
        megatron_conf['clip-grad'] = ds_config_conf.get('gradient_clipping', GRADIENT_CLIPPING_DEFAULT)
        _set_scheduler_params(ds_config_conf, megatron_conf)
        _set_optimizer_params(ds_config_conf, megatron_conf)

        return ds_runner_conf, megatron_conf, ds_config_conf

    @staticmethod
    def convert_to_old_args(args, parsed_args, ds_runner_conf, megatron_conf, ds_config_conf):
        """
        Split configuration into DS runner, megatron and DS conf file parts.
        Convert constituents into arguments which deepspeed and megatron expect.
        """

        def convert_(k, v):
            if isinstance(v, bool):
                if v:
                    return [f'--{k}']
                else:
                    return []
            if v is None:
                return []
            return [f'--{k}', str(v)]

        # Convert to CLI args
        ds_runner_args = [e for k, v in ds_runner_conf.items() for e in convert_(k, v)]
        user_script_args = (
                [e for k, v in megatron_conf.items() for e in convert_(k, v)]
                + ['--deepspeed_config', json.dumps(ds_config_conf, separators=(',', ':'))])

        old_style_args = ds_runner_args + [parsed_args.user_script] + user_script_args

        return old_style_args

    def consume_args(self, args=None, extra_conf=None):
        """
        Parse CLI args. Transform and derive other params.
        Convert to old style CLI args for deepspeed and megatron.
        """
        parser = self.construct_arg_parser()
        parsed_args, conf = self.parse_args(parser, args, extra_conf)
        ds_runner_conf, megatron_conf, ds_config_conf = self.derive_params_and_split(conf)
        old_style_args = self.convert_to_old_args(args, parsed_args, ds_runner_conf, megatron_conf, ds_config_conf)
        log.info(f"GPT-NEOX config: {conf}")
        return old_style_args, conf
