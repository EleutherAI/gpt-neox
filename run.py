#!/usr/bin/env python

# Variant of deepy.py that only parses configs and runs the user script.
# Intended to be used in enviroments where multinode launching is handled
# separately (e.g. directly through slurm).

import sys
import subprocess

from logging import warning

from megatron.neox_arguments import NeoXArgs


def strip_launcher_args(args):
    # Assumes that get_deepspeed_main_args places launcher args first
    # and then the user script followed by --deepspeed_config and
    # later --megatron_config.
    idx = args.index('--deepspeed_config') - 1
    launcher_args, remaining_args = args[:idx], args[idx:]
    assert '--megatron_config' in remaining_args    # sanity

    if launcher_args:
        warning(f'dropping launcher args: {launcher_args}')

    return remaining_args


def main(argv):
    all_args = NeoXArgs.consume_deepy_args()
    main_args = all_args.get_deepspeed_main_args()
    args = strip_launcher_args(main_args)

    result = subprocess.Popen(['python'] + args)
    result.wait()


if __name__ == '__main__':
    main(sys.argv)
