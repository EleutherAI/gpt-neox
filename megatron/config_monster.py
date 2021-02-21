import argparse
import os
import yaml

ds_runner_keys = ['hostfile', 'include', 'exclude', 'num_nodes', 'num_gpus', 'master_port', 'master_addr', 'launcher',
                  'launcher_args', 'user_script', 'user_args']
megatron_keys = ['num_layers', 'num_unique_layers', 'param_sharing_style', 'hidden_size', 'num_attention_heads',
                 'max_position_embeddings', 'make_vocab_size_divisible_by', 'layernorm_epsilon',
                 'apply_residual_connection_post_layernorm', 'openai_gelu', 'onnx_safe', 'attention_dropout',
                 'hidden_dropout', 'weight_decay', 'clip_grad', 'adam_beta1', 'adam_beta2', 'adam_eps', 'batch_size',
                 'onebitadam', 'gas', 'checkpoint_activations', 'distribute_checkpointed_activations',
                 'checkpoint_num_layers', 'train_iters', 'log_interval', 'exit_interval', 'tensorboard_dir',
                 'scaled_upper_triang_masked_softmax_fusion', 'scaled_masked_softmax_fusion', 'bias_gelu_fusion',
                 'geglu', 'no_weight_tying', 'sinusoidal_pos_emb', 'bias_dropout_fusion', 'sparsity', 'cpu_optimizer',
                 'cpu_torch_adam', 'seed', 'init_method_std', 'lr', 'lr_decay_style', 'lr_decay_iters', 'min_lr',
                 'warmup', 'override_lr_scheduler', 'use_checkpoint_lr_scheduler', 'save', 'save_interval',
                 'no_save_optim', 'no_save_rng', 'load', 'no_load_optim', 'no_load_rng', 'finetune',
                 'apply_query_key_layer_scaling', 'attention_softmax_in_fp32', 'fp32_allreduce', 'hysteresis',
                 'loss_scale', 'loss_scale_window', 'min_scale', 'fp16_lm_cross_entropy', 'model_parallel_size',
                 'pipe_parallel_size', 'distributed_backend', 'DDP_impl', 'local_rank', 'lazy_mpu_init',
                 'use_cpu_initialization', 'eval_iters', 'eval_interval', 'data_path', 'split', 'vocab_file',
                 'merge_file', 'seq_length', 'mask_prob', 'short_seq_prob', 'mmap_warmup', 'num_workers',
                 'tokenizer_type', 'data_impl', 'reset_position_ids', 'reset_attention_mask', 'eod_mask_loss',
                 'adlr_autoresume', 'adlr_autoresume_interval', 'ict_head_size', 'ict_load', 'bert_load',
                 'titles_data_path', 'query_in_block_prob', 'use_one_sent_docs', 'report_topk_accuracies',
                 'faiss_use_gpu', 'block_data_path', 'indexer_batch_size', 'indexer_log_interval', 'zero_stage',
                 'zero_reduce_scatter', 'zero_contigious_gradients', 'zero_reduce_bucket_size',
                 'zero_allgather_bucket_size', 'deepspeed_activation_checkpointing', 'partition_activations',
                 'contigious_checkpointing', 'checkpoint_in_cpu', 'synchronize_each_layer', 'profile_backward',
                 'deepspeed', 'deepspeed_config', 'deepscale', 'deepscale_config', 'deepspeed_mpi']
ds_config_keys = ['train_batch_size', 'train_micro_batch_size_per_gpu', 'steps_per_print', 'optimizer',
                  'gradient_clipping', 'fp16', 'wall_clock_breakdown', 'zero_allow_untested_optimizer']

ds_runner_keys_exclude = []
megatron_keys_exclude = [
    'fp16',  # Duplicated in ds_config
]
ds_config_keys_exclude = []


class ConfigMonster:
    """ Clearing up megatron's config monstrosity """

    def __init__(self):
        pass

    def construct_arg_parser(self):
        parser = argparse.ArgumentParser(description='GPT-NEOX Configuration',
                                         allow_abbrev=False)

        parser.add_argument("user_script",
                            type=str,
                            help="User script to launch, followed by any required "
                                 "arguments.")

        parser.add_argument("--conf_dir",
                            aliases=['-d'],
                            type=str,
                            default=None,
                            help="Directory to prefix to all configuration file paths")

        parser.add_argument("conf",
                            type=str,
                            nargs='+',
                            help="Configuration file path. Multiple files can be provided and will be merged.")

    def parse_args(self, args=None):
        parser = self.construct_arg_parser()
        args = parser.parse_args(args)

        # Validate user_script exists
        assert os.path.exists(args.user_script), f"User script could not be found: {args.user_script}"

        conf_files = args.conf
        if args.conf_dir:
            conf_files = [os.path.join(args.conf_dir, f) for f in conf_files]

        # Load and merge all configuration
        conf = {}
        for f in conf_files:
            with open(f) as f:
                conf_i = yaml.load(f)

            # Check there is are no duplicate keys
            confs_keys = set(conf.keys())
            conf_i_keys = set(conf_i.keys())
            assert confs_keys.intersection(conf_i_keys) == 0, f'Conf file {f} has duplicate keys with previously ' \
                                                              f'loaded file. '

            conf.update(conf_i)

        return args, conf

    def split_conf(self, args, conf):
        """ Split configuration into DS runner, megatron and DS conf file parts. """
        pass

