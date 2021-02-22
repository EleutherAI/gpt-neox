import argparse
import json
import os
import shlex
import sys
import logging
import yaml

log = logging.getLogger('ConfigMonster')

ds_runner_keys = ['hostfile', 'include', 'exclude', 'num_nodes', 'num_gpus', 'master_port', 'master_addr', 'launcher',
                  'launcher_args']  # handle separately: 'user_script', 'user_args'
megatron_keys = ['num-layers', 'num-unique-layers', 'param-sharing-style', 'hidden-size', 'num-attention-heads',
                 'max-position-embeddings', 'make-vocab-size-divisible-by', 'layernorm-epsilon',
                 'apply-residual-connection-post-layernorm', 'openai-gelu', 'onnx-safe', 'attention-dropout',
                 'hidden-dropout', 'weight-decay', 'clip-grad', 'adam-beta1', 'adam-beta2', 'adam-eps', 'batch-size',
                 'onebitadam', 'gas', 'checkpoint-activations', 'distribute-checkpointed-activations',
                 'checkpoint-num-layers', 'train-iters', 'log-interval', 'exit-interval', 'tensorboard-dir',
                 'scaled-upper-triang-masked-softmax-fusion', 'scaled-masked-softmax-fusion', 'bias-gelu-fusion',
                 'geglu', 'no-weight-tying', 'sinusoidal-pos-emb', 'bias-dropout-fusion', 'sparsity', 'cpu-optimizer',
                 'cpu_torch_adam', 'seed', 'init-method-std', 'lr', 'lr-decay-style', 'lr-decay-iters', 'min-lr',
                 'warmup', 'override-lr-scheduler', 'use-checkpoint-lr-scheduler', 'save', 'save-interval',
                 'no-save-optim', 'no-save-rng', 'load', 'no-load-optim', 'no-load-rng', 'finetune',
                 'apply-query-key-layer-scaling', 'attention-softmax-in-fp32', 'fp32-allreduce', 'hysteresis',
                 'loss-scale', 'loss-scale-window', 'min-scale', 'fp16-lm-cross-entropy', 'model-parallel-size',
                 'pipe-parallel-size', 'distributed-backend', 'DDP-impl', 'local_rank', 'lazy-mpu-init',
                 'use-cpu-initialization', 'eval-iters', 'eval-interval', 'data-path', 'split', 'vocab-file',
                 'merge-file', 'seq-length', 'mask-prob', 'short-seq-prob', 'mmap-warmup', 'num-workers',
                 'tokenizer-type', 'data-impl', 'reset-position-ids', 'reset-attention-mask', 'eod-mask-loss',
                 'adlr-autoresume', 'adlr-autoresume-interval', 'ict-head-size', 'ict-load', 'bert-load',
                 'titles-data-path', 'query-in-block-prob', 'use-one-sent-docs', 'report-topk-accuracies',
                 'faiss-use-gpu', 'block-data-path', 'indexer-batch-size', 'indexer-log-interval', 'zero-stage',
                 'zero-reduce-scatter', 'zero-contigious-gradients', 'zero-reduce-bucket-size',
                 'zero-allgather-bucket-size', 'deepspeed-activation-checkpointing', 'partition-activations',
                 'contigious-checkpointing', 'checkpoint-in-cpu', 'synchronize-each-layer', 'profile-backward',
                 'deepspeed', 'deepspeed_config', 'deepscale', 'deepspeed_mpi'] # 'fp16' is duplicate

ds_config_keys = ['train_batch_size', 'train_micro_batch_size_per_gpu', 'steps_per_print', 'optimizer',
                  'gradient_clipping', 'fp16', 'wall_clock_breakdown', 'zero_allow_untested_optimizer']
neox_config_keys = ['wandb_group', 'wandb_team']

ds_runner_keys_exclude = []
megatron_keys_exclude = [
    'fp16',  # Duplicated in ds_config
]
ds_config_keys_exclude = []


class ConfigMonster:
    """ Clearing up megatron's config monstrosity. """

    def __init__(self):
        pass

    def construct_arg_parser(self):
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

    def parse_args(self, parser: argparse.ArgumentParser, args=None, extra_conf=None):
        args = parser.parse_args(args)

        # Validate user_script exists
        assert os.path.exists(args.user_script), f"User script could not be found: {args.user_script}"

        conf_files = args.conf_file
        if args.conf_dir:
            conf_files = [os.path.join(args.conf_dir, f) for f in conf_files]

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
            log.info(f'Configuration parameters not specified: {", ".join(params_missing)}')

        return args, conf

    def derive_params_and_split(self, conf):
        """ Derive and insert implicit parameters """

        ds_runner_conf = {key: conf[key] for key in ds_runner_keys if key in conf}
        megatron_conf = {key: conf[key] for key in megatron_keys + neox_config_keys if key in conf}
        ds_config_conf = {key: conf[key] for key in ds_config_keys if key in conf}

        # fp16 is duplicated from DS runner config
        megatron_conf['fp16'] = conf.get('fp16', {}).get('enabled', False)

        return ds_runner_conf, megatron_conf, ds_config_conf

    def convert_to_old_args(self, args, parsed_args, ds_runner_conf, megatron_conf, ds_config_conf):
        """
        Split configuration into DS runner, megatron and DS conf file parts.
        Convert constituents into arguments which deepspeed and megatron expect.
        """

        def convert_(k, v):
            if v == True:
                return [f'--{k}']
            if v is None:
                return []
            return [f'--{k}', shlex.quote(str(v))]

        # Convert to CLI args
        ds_runner_args = [e for k, v in ds_runner_conf.items() for e in convert_(k, v)]
        user_script_args = (
                [e for k, v in megatron_conf.items() for e in convert_(k, v)]
                + ['--deepspeed_config', shlex.quote(json.dumps(ds_config_conf))])

        old_style_args = ds_runner_args + [parsed_args.user_script] + user_script_args

        return old_style_args

    def consume_args(self, args=None, extra_conf=None):
        """ Parse CLI args. Transform and derive other params.
        Convert to old style CLI args for deepspeed and megatron. """
        parser = self.construct_arg_parser()
        parsed_args, conf = self.parse_args(parser, args, extra_conf)
        ds_runner_conf, megatron_conf, ds_config_conf = self.derive_params_and_split(conf)
        old_style_args = self.convert_to_old_args(args, parsed_args, ds_runner_conf, megatron_conf, ds_config_conf)
        return old_style_args
