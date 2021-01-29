import os
import tarfile
import argparse
import deepspeed
import json
from collections import defaultdict
import mpu
import torch
import random 
import numpy as np
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP



# helpers
def get_args():
    parser = argparse.ArgumentParser(description='GPTNeox Deepspeed Training Script')
    # Include DeepSpeed configuration arguments
    parser.add_argument('--model', type=str, default="gpt3_small")
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def get_params(model):
    model_path = model if model.endswith(".json") else f"./configs/{model}.json"
    with open(model_path) as f:
        params = json.load(f)
    return defaultdict(lambda: None, params)


def is_main(args):
    """
    returns True if process is being run on the main GPU
    """
    return args.local_rank in [0, -1]


def get_all_files(filetype, files_dir):
    files = []
    for (dir_path, _, filenames) in os.walk(files_dir):
        for filename in filenames:
            if filename.endswith(".{}".format(filetype)):
                file_path = os.path.join(dir_path, filename)
                files.append(file_path)
    return files


def extract_tarfile(tarfile_path, extract_dir=None):
    dataset_tar = tarfile.open(tarfile_path, "r:gz")
    os.makedirs(extract_dir, exist_ok=True)
    dataset_tar.extractall(extract_dir)
    dataset_tar.close()


def cycle(loader):
    while True:
        for data in loader:
            yield data


def decode_token(token):
    return str(chr(max(32, token)))


def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))


def prepare_optimizer_parameters(model):
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    weight_decay = 0.01
    optimizer_grouped_parameters = [{
        'params':
            [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
            weight_decay
    }, {
        'params':
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
            0.0
    }]
    return optimizer_grouped_parameters


class DictArgs(dict):
    def __init__(self, config):
        for k, v in config.items():
            self[k] = v

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

#from https://github.com/microsoft/DeepSpeedExamples/blob/400cd1bc3524507301f39c659d3069672c4ab464/Megatron-LM/utils.py#L271
def get_checkpoint_name(checkpoints_path, iteration, release=False, zero=False):
    if release:
        d = 'release'
    else:
        d = 'iter_{:07d}'.format(iteration)
    if zero:
        dp_rank = mpu.get_data_parallel_rank()
        d += '_zero_dp_rank_{}'.format(dp_rank)
    return os.path.join(checkpoints_path, d,
                        'mp_rank_{:02d}'.format(mpu.get_model_parallel_rank()),
                        'model_optim_rng.pt')


def ensure_directory_exists(filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_checkpoint_tracker_filename(checkpoints_path):
    return os.path.join(checkpoints_path, 'latest_checkpointed_iteration.txt')


def save_zero_checkpoint(args, iteration, optimizer):
    zero_sd = {'iteration': iteration,
               'optimizer_state_dict': optimizer.state_dict()}
    zero_checkpoint_name = get_checkpoint_name(args.save, iteration, zero=True)
    ensure_directory_exists(zero_checkpoint_name)
    torch.save(zero_sd, zero_checkpoint_name)
    print('  successfully saved {}'.format(zero_checkpoint_name))

def save_checkpoint(iteration, model, optimizer,
                    lr_scheduler, args):
    """Save a model checkpoint."""
    if args.deepspeed:
        save_ds_checkpoint(iteration, model, args)
    else:
        # Only rank zer0 of the data parallel writes to the disk.
        if isinstance(model, torchDDP):
            model = model.module

        if mpu.get_data_parallel_rank() == 0:
            checkpoint_name = get_checkpoint_name(args.save, iteration)
            print('global rank {} is saving checkpoint at iteration {:7d} to {}'.
                format(torch.distributed.get_rank(), iteration, checkpoint_name))

            sd = {}
            sd['iteration'] = iteration
            sd['model'] = model.state_dict()

            # Optimizer stuff.
            if not args.no_save_optim:
                if optimizer is not None:
                    sd['optimizer'] = optimizer.state_dict()
                if lr_scheduler is not None:
                    sd['lr_scheduler'] = lr_scheduler.state_dict()

            # rng states.
            if not args.no_save_rng:
                sd['random_rng_state'] = random.getstate()
                sd['np_rng_state'] = np.random.get_state()
                sd['torch_rng_state'] = torch.get_rng_state()
                sd['cuda_rng_state'] = torch.cuda.get_rng_state()
                sd['rng_tracker_states'] = mpu.get_cuda_rng_tracker().get_states()


            ensure_directory_exists(checkpoint_name)
            torch.save(sd, checkpoint_name)
            print('  successfully saved {}'.format(checkpoint_name))

    # Wait so everyone is done (necessary)
    torch.distributed.barrier()
    # And update the latest iteration
    if torch.distributed.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, 'w') as f:
            f.write(str(iteration))
    # Wait so everyone is done (not necessary)
    torch.distributed.barrier()

def save_ds_checkpoint(iteration, model, args):
    """Save a model checkpoint."""

    sd = {}
    sd['iteration'] = iteration
    # rng states.
    if not args.no_save_rng:
        sd['random_rng_state'] = random.getstate()
        sd['np_rng_state'] = np.random.get_state()
        sd['torch_rng_state'] = torch.get_rng_state()
        sd['cuda_rng_state'] = torch.cuda.get_rng_state()
        sd['rng_tracker_states'] = mpu.get_cuda_rng_tracker().get_states()
        
    model.save_checkpoint(args.save, iteration, client_state = sd)


def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def get_checkpoint_iteration(args):
    # Read the tracker file and set the iteration.
    tracker_filename = get_checkpoint_tracker_filename(args.load)
    if not os.path.isfile(tracker_filename):
        print_rank_0('WARNING: could not find the metadata file {} '.format(
            tracker_filename))
        print_rank_0('    will not load any checkpoints and will start from '
                     'random')
        return 0, False, False
    iteration = 0
    release = False
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            release = metastring == 'release'
            if not release:
                print_rank_0('ERROR: Invalid metadata file {}. Exiting'.format(
                    tracker_filename))
                exit()

    assert iteration > 0 or release, 'error parsing metadata file {}'.format(
        tracker_filename)
    
    return iteration, release, True

def load_checkpoint(model, optimizer, lr_scheduler, args):
    """Load a model checkpoint."""

    iteration, release, success = get_checkpoint_iteration(args)

    if not success:
        return 0
        
    if args.deepspeed:

        checkpoint_name, sd = model.load_checkpoint(args.load, iteration)

        if checkpoint_name is None:
            if mpu.get_data_parallel_rank() == 0:
                print("Unable to load checkpoint.")
            return iteration

    else:
        
        # Checkpoint.
        checkpoint_name = get_checkpoint_name(args.load, iteration, release)
        
        if mpu.get_data_parallel_rank() == 0:
            print('global rank {} is loading checkpoint {}'.format(
                torch.distributed.get_rank(), checkpoint_name))

        # Load the checkpoint.
        sd = torch.load(checkpoint_name, map_location='cpu')

        if isinstance(model, torchDDP):
            model = model.module
        
        # Model.
        try:
            model.load_state_dict(sd['model'])
        except KeyError:
            print_rank_0('A metadata file exists but unable to load model '
                        'from checkpoint {}, exiting'.format(checkpoint_name))
            exit()

        # Optimizer.
        if not release and not args.finetune and not args.no_load_optim:
            try:
                if optimizer is not None:
                    optimizer.load_state_dict(sd['optimizer'])
                if lr_scheduler is not None:
                    lr_scheduler.load_state_dict(sd['lr_scheduler'])
            except KeyError:
                print_rank_0('Unable to load optimizer from checkpoint {}, exiting. '
                            'Specify --no-load-optim or --finetune to prevent '
                            'attempting to load the optimizer '
                            'state.'.format(checkpoint_name))
                exit()

    # Iterations.
    if args.finetune or release:
        iteration = 0
    else:
        try:
            iteration = sd['iteration']
        except KeyError:
            try: # Backward compatible with older checkpoints
                iteration = sd['total_iters']
            except KeyError:
                print_rank_0('A metadata file exists but Unable to load iteration '
                             ' from checkpoint {}, exiting'.format(checkpoint_name))
                exit()
                
    # rng states.
    if not release and not args.finetune and not args.no_load_rng:
        try:
            random.setstate(sd['random_rng_state'])
            np.random.set_state(sd['np_rng_state'])
            torch.set_rng_state(sd['torch_rng_state'])
            torch.cuda.set_rng_state(sd['cuda_rng_state'])
            mpu.get_cuda_rng_tracker().set_states(sd['rng_tracker_states'])
        except KeyError:
            print_rank_0('Unable to load optimizer from checkpoint {}, exiting. '
                         'Specify --no-load-optim or --finetune to prevent '
                         'attempting to load the optimizer '
                         'state.'.format(checkpoint_name))
            exit()

    torch.distributed.barrier()
    if mpu.get_data_parallel_rank() == 0:
        print('  successfully loaded {}'.format(checkpoint_name))

    return iteration