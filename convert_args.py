#!/usr/bin/env python

import json
import argparse

import deepspeed
from deepspeed.launcher.runner import parse_args as deepspeed_runner_parse_args

from megatron.arguments import _add_network_size_args, _add_regularization_args, _add_training_args, \
    _add_initialization_args, _add_learning_rate_args, _add_checkpointing_args, _add_mixed_precision_args, \
    _add_distributed_args, _add_validation_args, _add_data_args, _add_autoresume_args, _add_realm_args, _add_zero_args, \
    _add_activation_checkpoint_args


def megatron_parse_args(args, extra_args_provider=None, defaults={}, ignore_unknown_args=False):
    # FIRST LINES FROM FUNCTION: megatron.arguments.parse_args
    """Parse all arguments."""
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
    parser = _add_realm_args(parser)
    parser = _add_zero_args(parser)
    parser = _add_activation_checkpoint_args(parser)

    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    # Parse.
    if ignore_unknown_args:
        args, _ = parser.parse_known_args(args)
    else:
        args = parser.parse_args(args)
    return args

print('DS RUNNER ARGS')
ds_runner_args = deepspeed_runner_parse_args()
ds_runner_args = vars(ds_runner_args)
ds_runner_args_json = json.dumps(ds_runner_args)
print(ds_runner_args_json)



print('MEGATRON ARGS')
user_script_args = ds_runner_args['user_args']
megatron_args = megatron_parse_args(user_script_args)
megatron_args = vars(megatron_args)
megatron_args_json = json.dumps(megatron_args)
print(megatron_args_json)


print('DS CONFIG ARGS')
with open(megatron_args['deepspeed_config']) as f:
    ds_config = json.load(f)
with open(megatron_args['deepspeed_config']) as f:
    ds_config_json = f.read()
print(ds_config_json)

print('DS RUNNER KEYS')
print(list(ds_runner_args.keys()))

print('MEGATRON KEYS')
print(list(megatron_args.keys()))

print('DS CONFIG KEYS')
print(list(ds_config.keys()))


