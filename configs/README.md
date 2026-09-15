# Configuration and parameters

GPT-NeoX parameters are defined in a YAML configuration file which is passed to the `deepy.py` launcher - for examples see the files contained in this folder.
Parameters originate from either the [DeepSpeed runner CLI (DSL)](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/launcher/runner.py#L33), [DeepSpeed configuration file (DSC)](https://www.deepspeed.ai/docs/config-json/), [Megatron-LM CLI (Meg)](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/arguments.py#L224) or are GPT-NeoX (NeoX) modifications.

## Example Configuration (GPT3 Small):

Below is an example configuration `.yaml` to train a ~160M parameter GPT model. This readme will go through each section in the configuration and the options available.

For a detailed list of all the arguments available for neox, see [neox_arguments.md](neox_arguments.md)

Note: yaml arguments may be formatted with either '-' or '\_'. The standard separator used is a '\_' as shown in the example configurations below. However, the use of '-' as a separator may be deprecated in the future.
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
    # alternatively, use train_epochs to automatically determine the number of training iterations
    #"train_epochs": 1,
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

For KTO style training, you'll need to add the reward & label data path, e.g.:

```yaml
   "data_impl": "mmap",
   # Suggested data paths when using GPT-NeoX locally
   "train_data_path": "data/enwik8/enwik8_text_document",
   "train_label_data_path": "data/enwik8/enwik8_text_label_document",
   "train_reward_data_path": "data/enwik8/enwik8_text_reward_document",
   "test_data_path": "data/enwik8/enwik8_text_document",
   "test_label_data_path": "data/enwik8/enwik8_text_label_document",
   "test_reward_data_path": "data/enwik8/enwik8_text_reward_document",
   "valid_data_path": "data/enwik8/enwik8_text_document",
   "valid_label_data_path": "data/enwik8/enwik8_text_label_document",
   "valid_reward_data_path": "data/enwik8/enwik8_text_reward_document",
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

For DPO style training, you'll need to set pos/neg data paths instead of a single one, e.g.

```yaml
   "dataset_impl": "pairwise",
   "train_impl": "dpo",
   "pack_impl": "unpacked",
   "dpo_beta": 0.1,
   "dpo_fp32": true,
   "pos_train_data_path": "data/enwik8/enwik8_text_pos_document",
   "pos_valid_data_path": "data/enwik8/enwik8_text_pos_document",
   "pos_test_data_path": "data/enwik8/enwik8_text_pos_document",
   "neg_train_data_path": "data/enwik8/enwik8_text_neg_document",
   "neg_valid_data_path": "data/enwik8/enwik8_text_neg_document",
   "neg_test_data_path": "data/enwik8/enwik8_text_neg_document",
   ## If you have labels... (likely to mask out user turns)
   "pos_train_label_data_path": "data/enwik8/enwik8_text_pos_label_document",
   "pos_valid_label_data_path": "data/enwik8/enwik8_text_pos_label_document",
   "pos_test_label_data_path": "data/enwik8/enwik8_text_pos_label_document",
   "neg_train_label_data_path": "data/enwik8/enwik8_text_neg_label_document",
   "neg_valid_label_data_path": "data/enwik8/enwik8_text_neg_label_document",
   "neg_test_label_data_path": "data/enwik8/enwik8_text_neg_label_document",
   ## If you want to precompute the logits over your dataset...
   "precompute_model_name": "gpt2",
   ## Needed for the generation.py step, if precomputing
   "text_gen_type": "precompute"
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

### Mixed Precision Training Settings

Training precision can be selected with the top-level `precision` setting:

```yaml
   "precision": "fp16",
```

The supported values are `fp16`, `bfloat16`, and `fp32`. For compatibility with
existing configs, `precision` may instead be omitted and the corresponding
dictionary enabled directly:

```yaml
   "fp16": {
     "enabled": true,
     "loss_scale_window": 1000
   },
```

In that form GPT-NeoX derives `precision: fp16`; `bf16.enabled: true` similarly
derives `precision: bfloat16`. Explicit overrides—even when equal to a default—
are preserved. GPT-NeoX fills in any omitted values below. Only the listed keys
are accepted.

#### `fp16` dictionary

| Key | Type | Effective default | What it does |
| --- | --- | --- | --- |
| `enabled` | boolean | `true` | Enables DeepSpeed FP16. GPT-NeoX supplies `true` when `precision: fp16`; set it explicitly when selecting FP16 without `precision`. |
| `auto_cast` | boolean | `false` | Recursively casts floating-point inputs passed through the DeepSpeed engine to FP16 before the model forward pass. It does not select the model precision. |
| `loss_scale` | number | `0` | Sets the loss scale. `0` selects dynamic loss scaling; a nonzero value selects a fixed, static loss scale. |
| `initial_scale_power` | integer | `16` | Starts dynamic loss scaling at `2 ** initial_scale_power` (by default, 65,536). Ignored with a static `loss_scale`. |
| `loss_scale_window` | integer | `1000` | Number of overflow-free optimizer steps before the dynamic scale is doubled. Ignored with a static `loss_scale`. |
| `hysteresis` | integer | `2` | Number of overflow events required before the dynamic scale is halved. Ignored with a static `loss_scale`; see the optimizer-path note below. |
| `consecutive_hysteresis` | boolean | `false` | If `true`, a non-overflowing step resets the hysteresis counter, so the overflows must be consecutive to reduce the scale. If `false`, the counter resets when the scale increases after a stable window. Ignored with a static `loss_scale`; see below. |
| `min_loss_scale` | number | `1` | Lower bound for the dynamic loss scale. Ignored with a static `loss_scale`. |
| `fp16_master_weights_and_grads` | boolean | `false` | Keeps master weights and gradients in FP16 while optimizer states remain FP32. DeepSpeed supports this only with ZeRO stage 2, optimizer offload, and `DeepSpeedCPUAdam`; other combinations fail validation. |

The dynamic scaler's increase and decrease factor is fixed at 2 and is not a
dictionary option. `hysteresis` and `consecutive_hysteresis` are honored by the
loss scaler used with ZeRO stages 1–3. DeepSpeed's non-ZeRO fused and unfused
FP16 optimizer wrappers do not consume those two settings; they reduce the
scale on every overflow. The other dynamic settings are used by both paths.

For example, this keeps all defaults except the initial scale:

```yaml
   "precision": "fp16",
   "fp16": {
     "initial_scale_power": 12
   },
```

#### `bf16` dictionary

| Key | Type | Effective default | What it does |
| --- | --- | --- | --- |
| `enabled` | boolean | `true` | Enables DeepSpeed BF16. GPT-NeoX supplies `true` when `precision: bfloat16`; set it explicitly when selecting BF16 without `precision`. |
| `immediate_grad_update` | boolean | `false` | If `true`, DeepSpeed uses autograd hooks to transfer and accumulate BF16 gradients into the configured gradient-accumulation buffer (FP32 by default) as each gradient becomes available, instead of doing a bulk update after backward. This is used only when DeepSpeed selects its `BF16_Optimizer` wrapper; see below. |

DeepSpeed normally selects `BF16_Optimizer` without ZeRO. It also selects that
wrapper for ZeRO stage 1 when gradient accumulation is FP32 and optimizer CPU
offload is disabled. Other ZeRO paths do not consume `immediate_grad_update`.

BF16 does not use loss scaling: its effective loss scale is always 1. Therefore,
none of the FP16 loss-scaling keys, nor `auto_cast` or
`fp16_master_weights_and_grads`, may be placed in `bf16`. The minimal BF16
configuration can use either selector form:

```yaml
   "precision": "bfloat16",
```

or:

```yaml
   "bf16": {
     "enabled": true
   },
```

These keys and defaults match the EleutherAI DeeperSpeed revision pinned in
both dependency files. Do not put `type`, `fp16`, or `bf16` keys inside either
dictionary; use top-level `precision` or the dictionary's `enabled` key.


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
