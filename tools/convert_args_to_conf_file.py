#!/usr/bin/env python

import json
import argparse
import sys
from io import StringIO
from typing import Any
import dataclasses
import pandas as pd

import deepspeed
from dataclasses import dataclass
from deepspeed.constants import TORCH_DISTRIBUTED_DEFAULT_PORT
from deepspeed.launcher.constants import PDSH_LAUNCHER
from deepspeed.launcher.runner import DLTS_HOSTFILE

from megatron.arguments import _add_network_size_args, _add_regularization_args, _add_training_args, \
    _add_initialization_args, _add_learning_rate_args, _add_checkpointing_args, _add_mixed_precision_args, \
    _add_distributed_args, _add_validation_args, _add_data_args, _add_autoresume_args, _add_zero_args, \
    _add_activation_checkpoint_args

from megatron.config_monster import megatron_keys_exclude, ds_config_keys


@dataclass
class Param:
    name: str
    dest: str
    default: Any
    help: str


def get_deepspeed_runner_parser(args=None):
    ### MODIFIED FROM: deepspeed.launcher.runner.parse_args
    parser = argparse.ArgumentParser(
        description="DeepSpeed runner to help launch distributed "
                    "multi-node/multi-gpu training jobs.")

    parser.add_argument("-H",
                        "--hostfile",
                        type=str,
                        default=DLTS_HOSTFILE,
                        help="Hostfile path (in MPI style) that defines the "
                             "resource pool available to the job (e.g., "
                             "worker-0 slots=4)")

    parser.add_argument("-i",
                        "--include",
                        type=str,
                        default="",
                        help='''Specify hardware resources to use during execution.
                        String format is
                                NODE_SPEC[@NODE_SPEC ...],
                        where
                                NODE_SPEC=NAME[:SLOT[,SLOT ...]].
                        If :SLOT is omitted, include all slots on that host.
                        Example: -i "worker-0@worker-1:0,2" will use all slots
                        on worker-0 and slots [0, 2] on worker-1.
                        ''')

    parser.add_argument("-e",
                        "--exclude",
                        type=str,
                        default="",
                        help='''Specify hardware resources to NOT use during execution.
                        Mutually exclusive with --include. Resource formatting
                        is the same as --include.
                        Example: -e "worker-1:0" will use all available
                        resources except slot 0 on worker-1.
                        ''')

    parser.add_argument("--num_nodes",
                        type=int,
                        default=-1,
                        help="Total number of worker nodes to run on, this will use "
                             "the top N hosts from the given hostfile.")

    parser.add_argument("--num_gpus",
                        type=int,
                        default=-1,
                        help="Max number of GPUs to use on each node, will use "
                             "[0:N) GPU ids on each node.")

    parser.add_argument("--master_port",
                        default=TORCH_DISTRIBUTED_DEFAULT_PORT,
                        type=int,
                        help="(optional) Port used by PyTorch distributed for "
                             "communication during training.")

    parser.add_argument("--master_addr",
                        default="",
                        type=str,
                        help="(optional) IP address of node 0, will be "
                             "inferred via 'hostname -I' if not specified.")

    parser.add_argument("--launcher",
                        default=PDSH_LAUNCHER,
                        type=str,
                        help="(optional) choose launcher backend for multi-node"
                             "training. Options currently include PDSH, OpenMPI, MVAPICH.")

    parser.add_argument("--launcher_args",
                        default="",
                        type=str,
                        help="(optional) pass launcher specific arguments as a "
                             "single quoted argument.")

    parser.add_argument("user_script",
                        type=str,
                        help="User script to launch, followed by any required "
                             "arguments.")
    parser.add_argument('user_args', nargs=argparse.REMAINDER)
    return parser


def megatron_parse_args(args, extra_args_provider=None, defaults={}, ignore_unknown_args=False):
    # MODIFIED FROM: megatron.arguments.parse_args
    """Parse all arguments."""
    parser = get_megatron_parser(extra_args_provider)

    # Parse.
    if ignore_unknown_args:
        args, _ = parser.parse_known_args(args)
    else:
        args = parser.parse_args(args)
    return args


def get_megatron_parser(extra_args_provider=None):
    parser = argparse.ArgumentParser(description='Megatron-LM Arguments',
                                     allow_abbrev=False)
    # Standard arguments.
    parser = _add_network_size_args(parser)
    parser = _add_regularization_args(parser)
    parser = _add_training_args(parser)
    parser = _add_initialization_args(parser)
    parser = _add_learning_rate_args(parser)
    parser = _add_checkpointing_args(parser)
    parser = _add_mixed_precision_args(parser)
    parser = _add_distributed_args(parser)
    parser = _add_validation_args(parser)
    parser = _add_data_args(parser)
    parser = _add_autoresume_args(parser)
    parser = _add_zero_args(parser)
    parser = _add_activation_checkpoint_args(parser)
    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    return parser


def crude_arg_parser(args=sys.argv):
    """ Only consider optional args """
    args_dict = {}
    key = None
    for e in args[1:]:
        if e[:2] == '--':
            if key:
                args_dict[key] = True  # Switch arg
            key = e[2:]
        elif key:
            args_dict[key] = e
            key = None

    return args_dict


def parser_to_params_list(parser, exclude_args=None):
    params = [
        Param(e.option_strings[-1].split('--')[-1], e.dest, e.default, e.help)
        for e in parser._get_optional_actions()
        # Only consider optional arguments. You can get positional using `_get_positional_actions()`
    ]
    if exclude_args:
        params = [
            p for p in params if p.name not in exclude_args
        ]
    return params


def try_cast_to_number(e: Any):
    if isinstance(e, str):  # Could be a bool
        try:
            return int(e)
        except ValueError:
            pass
        try:
            return float(e)
        except ValueError:
            pass
    return e


args = crude_arg_parser()

print('Drop this into a deepspeed/megatron examples script instead of `deepspeed` to capture the arguments and '
      'produce a conf json')

print('------- Convert args to JSON -------')
conf_0 = crude_arg_parser()
converting_args = len(conf_0) > 0
if converting_args:
    conf = {k: try_cast_to_number(v) for k, v in conf_0.items() if k != 'deepspeed_config'}

    with open(conf_0['deepspeed_config']) as f:  # Get deepspeed bit. WARNING it will override keys
        ds_config = json.load(f)
    intersect = set(ds_config.keys()).intersection(set(conf.keys()))
    if len(intersect) > 0:
        print(f"WARNING KEYS OVERRIDDEN BY DS: {intersect}")
    conf.update(ds_config)

    conf_json = json.dumps(conf)
    print(conf_json)

print('------- DISCOVER ALL PARAMS -------')

print('--- DS RUNNER PARAMS ---')
ds_runner_parser = get_deepspeed_runner_parser()
ds_runner_params = parser_to_params_list(ds_runner_parser)
print([e.name for e in ds_runner_params])

print('--- MEGATRON PARAMS ---')
megatron_parser = get_megatron_parser()
megatron_parser_params = parser_to_params_list(megatron_parser, exclude_args=megatron_keys_exclude)
megatron_parser_params = [e for e in megatron_parser_params if e.name != 'deepscale_config']
print([e.name for e in megatron_parser_params])

print('--- DS CONFIG PARAMS FROM PROVIDED JSON (NOT ALL PARAMS) ---')
if converting_args:
    print(list(ds_config.keys()))

print('------- Document parameters -------')
info = pd.concat([
    pd.DataFrame(map(dataclasses.asdict, ds_runner_params)).assign(origin='DSR'),
    pd.DataFrame(map(dataclasses.asdict, megatron_parser_params)).assign(origin='Meg'),
    pd.DataFrame({'name': ds_config_keys, 'origin': 'DSC'}),
])
info = info[['origin', 'name', 'default', 'help']]

md_tb = StringIO()
info.to_csv(md_tb, index=False)
print(md_tb.getvalue())
