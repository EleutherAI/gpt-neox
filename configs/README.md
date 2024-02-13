# Configuration and parameters

GPT-NeoX parameters are defined in a YAML configuration file which is passed to the `deepy.py` launcher - for examples see the files contained in this folder.
Parameters originate from either the [DeepSpeed runner CLI (DSL)](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/launcher/runner.py#L33), [DeepSpeed configuration file (DSC)](https://www.deepspeed.ai/docs/config-json/), [Megatron-LM CLI (Meg)](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/arguments.py#L224) or are GPT-NeoX (NeoX) modifications.

## Example Configuration (GPT3 Small):

Below is an example configuration `.yaml` to train a ~160M parameter GPT model. This readme will go through each section in the configuration and the options available.

For a detailed list of all the arguments available for neox, see [neox_arguments.md](neox_arguments.md)

Note: yaml arguments may be formatted with either '-' or '_'. The standard separator used is a '_' as shown in the example configurations below. However, the use of '-' as a separator may be deprecated in the future.
```yaml
# GPT-3 pretraining setup
{
   # parallelism settings ( you will want to change these based on your cluster setup, ideally scheduling pipeline stages
   # across the node boundaries )
   "pipe_parallel_size": 1,
   "model_parallel_size": 1,

   # model settings
   "num_layers": 12,
   "hidden_size": 768,
   "num_attention_heads": 12,
   "seq_length": 2048,
   "max_position_embeddings": 2048,
   "norm": "rmsnorm",
   "pos_emb": "none",
   "no_weight_tying": true,
    # this should provide some speedup but takes a while to build, set to true if desired
   "scaled_upper_triang_masked_softmax_fusion": false,
   "train_iters": 320000,

   # optimizer settings
   "optimizer": {
     "type": "Adam",
     "params": {
       "lr": 0.0006,
       "max_grad_norm": 1.0,
       "betas": [0.9, 0.95]
     }
   },
   # for all zero_optimization options, see https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training
   "zero_optimization": {
    "stage": 0,
    "allgather_partitions": True,
    "allgather_bucket_size": 500000000,
    "overlap_comm": True,
    "reduce_scatter": True,
    "reduce_bucket_size": 500000000,
    "contiguous_gradients": True,
  },

   # batch / data settings
   "train_micro_batch_size_per_gpu": 4,
   "gradient_accumulation_steps": 1,
   "data_impl": "mmap",
   "split": "949,50,1",

   # activation checkpointing
   "checkpoint_activations": true,
   "checkpoint_num_layers": 1,
   "partition_activations": true,
   "synchronize_each_layer": true,

   # regularization
   "gradient_clipping": 1.0,
   "weight_decay": 0,
   "hidden_dropout": 0,
   "attention_dropout": 0,

   # precision settings
   "fp16": {
     "enabled": true,
     "loss_scale": 0,
     "loss_scale_window": 1000,
     "hysteresis": 2,
     "min_loss_scale": 1
   },

   # lr decay settings
   "lr_decay_iters": 320000,
   "lr_decay_style": "cosine",
   "warmup": 0.01,

   # misc. training settings
   "distributed_backend": "nccl",
   "checkpoint_factor": 10000,
   "eval_interval": 1000,
   "eval_iters": 10,

   # logging
   "log_interval": 100,
   "steps_per_print": 10,
   "keep_last_n_checkpoints": 4,
   "wall_clock_breakdown": true,
}
```

### Parallelism Settings:

The parallelism settings are left at 1 in all configs, as the settings you want will be highly dependent on your compute setup and network topology.
We have found it best to do model parallelism within a node, and schedule pipeline stages across node boundaries.

```yaml
   "pipe_parallel_size": 1,
   "model_parallel_size": 1,
```

These can be set to any integer between `0` and `num_gpus`, and `num_gpus` must be divisible by `pipe_parallel_size` * `model_parallel_size`.


### Model Settings:
```yaml
   # model settings
   "num_layers": 12,
   "hidden_size": 768,
   "num_attention_heads": 12,
   "seq_length": 2048,
   "max_position_embeddings": 2048,
   "norm": "rmsnorm",
   "pos_emb": "none",
   "no_weight_tying": true,
    # this should provide some speedup but takes a while to build, set to true if desired
   "scaled_upper_triang_masked_softmax_fusion": false,
   "train_iters": 320000,
```
An example of some basic settings used to configure your model's architecture and number of training steps.

### Optimizer Settings:

Our optimizer configuration has a similar syntax to deepspeed's. Different optimizers will have different arguments for "params".
Learning rate should be configured from here using the `"lr"` field of `optimizer["params"]`.

```yaml
  # optimizer settings
   "optimizer": {
     "type": "Adam",
     "params": {
       "lr": 0.0006,
       "max_grad_norm": 1.0,
       "betas": [0.9, 0.95]
     }
   }
   ```
Available optimizer types are:

- `"Adam"`: regular Adam optimizer
- `"OneBitAdam"`: Deepspeed's [OneBitAdam optimizer](https://www.deepspeed.ai/docs/config-json/#optimizer-parameters). To use 1-bit adam, you'll also need to add the `freeze_step`, `cuda_aware`, and `comm_backend_name` fields, like so:
```yaml
   "optimizer": {
     "type": "OneBitAdam",
     "params": {
       "lr": 0.0001,
       "freeze_step": 23000,
       "betas": [0.9, 0.95],
       "cuda_aware": false,
       "comm_backend_name": "nccl"
     }
```

- `"CPU_Adam"`/`"CPU_torch_adam"`: Adam optimizer on CPU. Either megatron's version ("CPU_Adam") or torch's ("CPU_torch_adam")
- `"SM3"`: SM3 or [Memory adaptive efficient optimization optimizer](https://arxiv.org/pdf/1901.11150.pdf). We have found this doesn't work well with fp16 training.
- `"madgrad_wd"`: MADGRAD or [A Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic
    Optimizer] weight decay has been implemented AdamW style instead of the original madgrad Adam style. https://arxiv.org/abs/2101.11075

### ZeRO Optimization:

```yaml
# for all zero_optimization options, see https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training
  "zero_optimization": {
        "stage": 0,
        "allgather_partitions": True,
        "allgather_bucket_size": 500000000,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 500000000,
        "contiguous_gradients": True,
  },
  "zero_allow_untested_optimizer": false,

```

ZeRO optimization in NeoX is currently configured identically to how deepspeed configures it, please see [the deepspeed docs](https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training) for more information.

If you want to combine an optimizer untested by DeepSpeed with ZeRO (i.e, not ADAM or LAMB), you must pass `"zero_allow_untested_optimizer": true` *outside* of the `"zero_optimization"` dictionary (see above).

N.B - ZeRO stages 2+ are incompatible with pipeline parallelism. Please set `"pipe-parallel-size"` to 0 if you want to use ZeRO stage 2 or more.

### Batch Size Settings:

```yaml
   # batch / data settings
   "train_micro_batch_size_per_gpu": 4,
   "gradient_accumulation_steps": 1,
```
Our global batch size configuration follows deepspeed's and can be configured in a number of ways. At least any one of `"train_batch_size"` and `"train_micro_batch_size_per_gpu"`.
- `"train_batch_size"`: The effective training batch size. This is the amount of data samples that leads to one step of model update. train_batch_size is aggregated by the batch size that a single GPU processes in one forward/backward pass (a.k.a., train_step_batch_size), the gradient accumulation steps (a.k.a., gradient_accumulation_steps), and the number of GPUs.
- `"train_micro_batch_size_per_gpu""`: Batch size to be processed by one GPU in one step (without gradient accumulation). When specified, `gradient_accumulation_steps` is automatically calculated using train_batch_size and number of GPUs.
- `"gradient_accumulation_steps"`: Number of training steps to accumulate gradients before averaging and applying them. This feature is sometimes useful to improve scalability since it results in less frequent communication of gradients between steps. Another impact of this feature is the ability to train with larger batch sizes per GPU. When specified, train_step_batch_size is automatically calculated using train_batch_size and number of GPUs.

### Extra DeepSpeed Settings

```yaml
# additional deepspeed args not specified above
"deepspeed_extra_args": {
    "comms_logger": {
        "enabled": true,
        "verbose": true,
        "prof_all": true,
        "debug": false
    },
}
```
Additional DeepSpeed settings besides those mentioned above should be wrapped in the `"deepspeed_extra_args` argument, as in the example above. This functionality is designed to allow arguments not specified by existing dataclasses to be passed to DeepSpeed (e.g. when new functionalities are implemented). If any settings are duplicated here from elsewhere in the YAML, the system will throw an exception and notify the user.

### Dataset / Tokenizer / Checkpoint / Logging Settings:

```yaml
   "data_impl": "mmap",
   "split": "949,50,1",
   # Suggested data paths when using GPT-NeoX locally
   "data_path": "data/enwik8/enwik8_text_document",
   #"train_data_path": "data/enwik8/enwik8_text_document",
   #"test_data_path": "data/enwik8/enwik8_text_document",
   #"valid_data_path": "data/enwik8/enwik8_text_document",
   "vocab_file": "data/gpt2-vocab.json",
   "merge_file": "data/gpt2-merges.txt",
   "save": "checkpoints",
   "load": "checkpoints",
   "tensorboard_dir": "tensorboard",
   "log_dir": "logs",
   "checkpoint_factor": 10000,
   "eval_interval": 1000,
   "eval_iters": 10,
```

### LR Scheduler settings

```yaml
   "lr_decay_iters": 320000,
   "lr_decay_style": "cosine",
   "warmup": 0.01,
```

Settings used to modify the learning rate over time.

N.B - `OneBitAdam` requires you to use deepspeed's internal lr scheduler because reasons. Currently the lr decay style defaults to deepspeed's `WarmupDecay

### Activation Checkpointing Settings:

```yaml
   "checkpoint_activations": true,
   "checkpoint_num_layers": 1,
   "partition_activations": true,
   "synchronize_each_layer": true,
```

Checkpointing works by trading compute for memory. Rather than storing all intermediate activations of the entire computation graph for computing backward, the checkpointed part does not save intermediate activations, and instead recomputes them in backward pass.

### Mixed Precision Training Settings:
gpt-neox's fp16 training is configured identically to DeepSpeed's, please see [their documentation](https://www.deepspeed.ai/docs/config-json/#fp16-training-options) for more information.
An example config for fp16 training:

```yaml
   "fp16": {
     "enabled": true,
     "loss_scale": 0,
     "loss_scale_window": 1000,
     "hysteresis": 2,
     "min_loss_scale": 1
   },
```

Alternatively you can use the `precision` config which can be set to `fp16`, `bfloat16`, or `fp32`. If you set `"precision": "fp16"` without adding a `"fp16": {...}` dict, then it will simply use DeepSpeed's defaults for fp16 training.


### SLURM Settings

If you are running GPT-NeoX on a SLURM cluster and wish to use SLURM to coordinate nodes, then you must set the following variables in your config:

```yaml
    "launcher": "slurm",
    "deepspeed_slurm": true
```

Additionally, you need to modify _all_ of your configs to conform to the JSON. When launching a GPT-NeoX job you can specify multiple YAML config files. Internally, all of these files are merged into one config and then passed as a single long command line argument to Deep(er)Speed. When using SLURM and its internal command `srun`, python fails to parse this long command line argument unless it is in the more restrictive JSON format. In practice, the example NeoX configs are already very close to JSON. As an example, this is a snippet of a YAML-compatible config, N.B. the comment the capital-F `False`:

```yaml
    # optimizer settings
   "optimizer": {
     "type": "OneBitAdam",
     "params": {
       "lr": 0.0001,
       "freeze_step": 23000,
       "betas": [0.9, 0.95],
       "cuda_aware": False,
       "comm_backend_name": "nccl"
     }
```

To make this JSON just remove the comment and use all lowercase for the boolean:

```yaml
   "optimizer": {
     "type": "OneBitAdam",
     "params": {
       "lr": 0.0001,
       "freeze_step": 23000,
       "betas": [0.9, 0.95],
       "cuda_aware": false,
       "comm_backend_name": "nccl"
     }
```
