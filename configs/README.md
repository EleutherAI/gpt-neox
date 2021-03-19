# Configuration and parameters

GPT-NeoX parameters are defined in a YAML configuration file which is passed to the `deepy.py` launcher - for examples see the files contained in this folder.
Parameters originate from either the [DeepSpeed runner CLI (DSL)](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/launcher/runner.py#L33), [DeepSpeed configuration file (DSC)](https://www.deepspeed.ai/docs/config-json/), [Megatron-LM CLI (Meg)](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/arguments.py#L224) or are GPT-NeoX (NeoX) modifications.

## Example Configuration (GPT3 Small):

Below is an example configuration `.yaml` to train a ~160M parameter GPT model. This readme will go through each section in the configuration and the options available.
```yaml
# GPT-3 pretraining setup
{
   # parallelism settings ( you will want to change these based on your cluster setup, ideally scheduling pipeline stages
   # across the node boundaries )
   "pipe-parallel-size": 1,
   "model-parallel-size": 1,

   # model settings
   "num-layers": 12,
   "hidden-size": 768,
   "num-attention-heads": 12,
   "seq-length": 2048,
   "max-position-embeddings": 2048,
   "norm": "rmsnorm",
   "pos-emb": "none",
   "no-weight-tying": true,
    # this should provide some speedup but takes a while to build, set to true if desired
   "scaled-upper-triang-masked-softmax-fusion": false,
   "train-iters": 320000,

   # optimizer settings
   "optimizer": {
     "type": "Adam",
     "params": {
       "lr": 0.0006,
       "max_grad_norm": 1.0,
       "betas": [0.9, 0.95]
     }
   },
   "zero_optimization": {
    "stage": 0,
    "allgather_partitions": True,
    "allgather_bucket_size": 500000000,
    "overlap_comm": True,
    "reduce_scatter": True,
    "reduce_bucket_size": 500000000,
    "contiguous_gradients": True,
    "cpu_offload": False
  },

   # batch / data settings
   "train_micro_batch_size_per_gpu": 4,
   "gradient_accumulation_steps": 1,
   "data-impl": "mmap",
   "split": "949,50,1",

   # activation checkpointing
   "checkpoint-activations": true,
   "checkpoint-num-layers": 1,
   "partition-activations": true,
   "synchronize-each-layer": true,

   # regularization
   "gradient_clipping": 1.0,
   "weight-decay": 0,
   "hidden-dropout": 0,
   "attention-dropout": 0,

   # precision settings
   "fp16": { 
     "enabled": true,
     "loss_scale": 0,
     "loss_scale_window": 1000,
     "hysteresis": 2,
     "min_loss_scale": 1
   },

   # lr decay settings
   "lr-decay-iters": 320000,
   "lr-decay-style": "cosine",
   "warmup": 0.01,
  
   # misc. training settings
   "distributed-backend": "nccl",
   "save-interval": 10000,
   "eval-interval": 1000,
   "eval-iters": 10,

   # logging
   "log-interval": 100,
   "steps_per_print": 10,
   "keep-last-n-checkpoints": 4,
   "wall_clock_breakdown": true,
}
```

### Parallelism Settings:

The parallelism settings are left at 1 in all configs, as the settings you want will be highly dependent on your compute setup and network topology.
We have found it best to do model parallelism within a node, and schedule pipeline stages across node boundaries.

```yaml
   "pipe-parallel-size": 1,
   "model-parallel-size": 1,
```

These can be set to any integer between `0` and `num_gpus`, and `num_gpus` must be divisible by `pipe_parallel_size` * `model_parallel_size`.

All Parallelism Settings are below:

| Origin | Parameter name                            | Default value | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|--------|-------------------------------------------|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Meg    | pipe-parallel-size                              |         0      | Number of pipeline parallel stages. Disable with 0.                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Meg    | model-parallel-size                                |       1        | Size of the model parallelism.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Meg    | pipe-partition-method                               |    "type:transformer"           | method used to distribute model layers across pipeline stages. Choose from "parameters", which balances the number of parameters on each pipeline stage, "uniform", which naively balances the number of layers per stage, or "type:[regex]" (in our case this will basically only be "type:transformer"), which balances layers whose class names match [regex]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        

### Model Settings:
```yaml
   # model settings
   "num-layers": 12,
   "hidden-size": 768,
   "num-attention-heads": 12,
   "seq-length": 2048,
   "max-position-embeddings": 2048,
   "norm": "rmsnorm",
   "pos-emb": "none",
   "no-weight-tying": true,
    # this should provide some speedup but takes a while to build, set to true if desired
   "scaled-upper-triang-masked-softmax-fusion": false,
   "train-iters": 320000,
```
An example of some basic settings used to configure your model's architecture and number of training steps. 

All available options for model settings are below:

| Origin | Parameter name                            | Default value | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|--------|-------------------------------------------|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Meg    | train-iters                               |               | Total number of iterations to train over all training runs.                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Meg    | num-layers                                |               | Number of transformer layers.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Meg    | hidden-size                               |               | Transformer hidden size.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Meg    | num-attention-heads                       |               | Number of transformer attention heads.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Meg    | seq-length                                |               | Maximum sequence length to process.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Meg    | max-position-embeddings                   |               | Maximum number of position embeddings to use. This is the size of position embedding.                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| NeoX   | norm                                      | layernorm     | Normalization layer to use. Choose from "layernorm", "rmsnorm" and "scalenorm".                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Meg    | layernorm-epsilon                         | 1e-05         | Layer norm epsilon.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| NeoX   | rms-norm-epsilon                          | 1e-8          | Root mean squared norm epsilon                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| NeoX   | scalenorm-epsilon                         | 1e-8          | Scalenorm epsilon                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| NeoX   | pos-emb                                   | learned       | Type of positional embedding to use - choose from 'learned', 'sinusoidal', 'rpe', 'none'                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| NeoX   | rpe-num-buckets                           | 32            | T5 relative positional encoding number of buckets, default 32.                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| NeoX   | rpe-max-distance                          | 128           | T5 relative positional encoding max distance, default 128.                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| NeoX   | no-weight-tying                           | false         | Disables weight tying between embedding weights and final Linear layer                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| NeoX   | geglu                                     | false         | Enable geglu activation function (WARNING: will increase memory usage, adjust embd dims accordingly)                                                                                                                                                                                                                                                                                                                                                                                                                           |
| NeoX   | sparsity                                  | none          | Sparse attention layer configuration: `none` = all regular attn, `all` = all sparse attn, `interspersed` = sparse on odd layers, dense on even.                                                                                                                                                                                                                                                                                                                                                                                |
| Meg    | num-unique-layers                         |               | Number of unique transformer layers. `num-layers` should be divisible by this value. Currently only has an effect when `pipe_parallel_size`=0.                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Meg    | param-sharing-style                       | grouped       | Ordering of the shared parameters. For example, for a `num-layers`=4 and `--num-unique-layers`=2, we will have the following ordering for two unique layers 1 and 2-: grouped: [1, 2, 1, 2] and spaced: [1, 1, 2, 2].                                                                                                                                                                                                                                                                                                          |
| Meg    | make-vocab-size-divisible-by              | 128           | Pad the vocab size to be divisible by this value. This is added for computational efficiency reasons.                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Meg    | apply-residual-connection-post-layernorm  | false         | If set, use original BERT residual connection ordering.                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Meg    | openai-gelu                               | false         | Use OpenAIs GeLU implementation. This option should not be used unless for backward compatibility reasons.                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Meg    | scaled-upper-triang-masked-softmax-fusion | false         | Enable fusion of query_key_value_scaling time (upper diagonal) masking and softmax.                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Meg    | scaled-masked-softmax-fusion              | false         | Enable fusion of query_key_value_scaling general masking and softmax.                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Meg    | bias-gelu-fusion                          | false         | Enable bias and gelu fusion.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Meg    | bias-dropout-fusion                       | false         | Enable bias and dropout fusion.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Meg    | fp16-lm-cross-entropy                     | false         | Move the cross entropy unreduced loss calculation for lm head to fp16.                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Meg    | init-method-std                           | 0.02          | Standard deviation of the zero mean normal distribution used for weight initialization.                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Meg    | apply-query-key-layer-scaling             | false         | Scale `Q * K^T` by `1 / layer-number`. If this flag is set, then it will automatically set attention-softmax-in-fp32 to `true`                                                                                                                                                                                                                                                                                                                                                                                                 |
| Meg    | use-cpu-initialization                    | false         | If set, affine parallel weights initialization uses CPU                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Meg    | attention-softmax-in-fp32                 | false         | Run attention masking and softmax in fp32.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Meg    | fp32-allreduce                            | false         | All-reduce in fp32                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |

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

### ZeRO Optimization:

```yaml
  "zero_optimization": {
        "stage": 0,
        "allgather_partitions": True,
        "allgather_bucket_size": 500000000,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 500000000,
        "contiguous_gradients": True,
        "cpu_offload": False
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

### Dataset / Tokenizer / Checkpoint / Logging Settings:

```yaml
   "data-impl": "mmap",
   "split": "949,50,1",
   # Suggested data paths when using GPT-NeoX locally
   "data-path": "data/enron/enron_text_document",
   "vocab-file": "data/gpt2-vocab.json",
   "merge-file": "data/gpt2-merges.txt",
   "save": "checkpoints",
   "load": "checkpoints",
   "tensorboard-dir": "tensorboard",
   "log-dir": "logs",
   "save-interval": 10000,
   "eval-interval": 1000,
   "eval-iters": 10,
```

These are the settings used to control the dataloading, train/test/val splits, tokenization and checkpointing. All available options for these settings are below:

| Origin | Parameter name                            | Default value | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|--------|-------------------------------------------|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Meg    | data-path                                 |               | Path to combined dataset to split.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| Meg    | data-impl                                 | infer         | Implementation of indexed datasets.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Meg    | mmap-warmup                               | false         | Warm up mmap files.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Meg    | save                                      |               | Output directory to save checkpoints to.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Meg    | load                                      |               | Directory containing a model checkpoint.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Meg    | save-interval                             |               | Number of iterations between checkpoint saves.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Meg    | seed                                      | 1234          | Random seed used for python, numpy, pytorch, and cuda.                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Meg    | no-save-optim                             | false         | Do not save current optimizer.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Meg    | no-save-rng                               | false         | Do not save current rng state.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Meg    | no-load-optim                             | false         | Do not load optimizer when loading checkpoint.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Meg    | no-load-rng                               | false         | Do not load rng state when loading checkpoint.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Meg    | finetune                                  | false         | Load model for finetuning. Do not load optimizer or rng state from checkpoint and set iteration to 0. Assumed when loading a release checkpoint.                                                                                                                                                                                                                                                                                                                                                                               |
| Meg    | eval-iters                                | 100           | Number of iterations to run for evaluationvalidation/test for.                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Meg    | eval-interval                             | 1000          | Interval between running evaluation on validation set.                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Meg    | split                                     | 969, 30, 1    | Comma-separated list of proportions for training, validation, and test split. For example the split `90,5,5` will use 90% of data for training, 5% for validation and 5% for test.                                                                                                                                                                                                                                                                                                                                             |
| Meg    | vocab-file                                |               | Path to the vocab file.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Meg    | merge-file                                |               | Path to the BPE merge file.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| NeoX   | log-dir                                   |               | Directory to save logs to.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| NeoX   | tensorboard-dir                           |               | Write TensorBoard logs to this directory.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Meg    | num-workers                               | 2             | Dataloader number of workers.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| DSC    | steps_per_print                           | 10            | Print train loss every N steps                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| DSC    | wall_clock_breakdown                      | false         | Enable timing of the latency of forward/backward/update training phases                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| DSC    | dump_state                                | false         | Print out state information of DeepSpeed object after initialization                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Meg    | tokenizer-type                            |               | What type of tokenizer to use. DEPRECATED - currently only GPT2Tokenizer is available.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Meg    | exit-interval                             |               | Exit the program after the iteration is divisible by this value.                                                                                                                                                                                                                                                                                                                                                                                                                                                               |

### LR Scheduler settings

```yaml
   "lr-decay-iters": 320000,
   "lr-decay-style": "cosine",
   "warmup": 0.01,
```

Settings used to modify the learning rate over time. All settings available:

| Origin | Parameter name                            | Default value | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|--------|-------------------------------------------|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Meg    | lr-decay-style                            | linear        | Learning rate decay function. Choose from 'constant', 'linear', 'cosine', 'exponential'.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Meg    | lr-decay-iters                            |               | Number of iterations to decay learning rate over, If None defaults to `--train-iters`                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Meg    | min-lr                                    | 0.0           | Minumum value for learning rate. The scheduler clips values below this threshold.                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Meg    | warmup                                    | 0.01          | Percentage of total iterations to warmup on (.01 = 1 percent of all training iters).                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Meg    | override-lr-scheduler                     | false         | Reset the values of the scheduler (learning rate,warmup iterations, minimum learning rate, maximum number of iterations, and decay style from input arguments and ignore values from checkpoints. Note that all the above values will be reset.                                                                                                                                                                                                                                                                                 |
| Meg    | use-checkpoint-lr-scheduler               | false         | Use checkpoint to set the values of the scheduler (learning rate, warmup iterations, minimum learning rate, maximum number of iterations, and decay style from checkpoint and ignore input arguments.                                                                                                                                                                                                                                                                                                                          |

N.B - OneBitAdam requires you to use deepspeed's internal lr scheduler because reasons. Currently the lr decay style defaults to deepspeed's `WarmupDecayLR`. min lr, lr and warmup should still be configured as above. We're working on making this more flexible.

### Regularization Settings: 

```yaml   
   "gradient_clipping": 1.0,
   "weight-decay": 0,
   "hidden-dropout": 0,
   "attention-dropout": 0,
```

Various settings to regularize the model. At larger scales, we find that regularization only slows down training and has little to negative effect. At smaller scales and on smaller datasets, however, it can improve performance.
All regularization settings below:

| Origin | Parameter name                            | Default value | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|--------|-------------------------------------------|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DSC    | gradient_clipping                         | 0             | Enable gradient clipping with provided value                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Meg    | attention-dropout                         | 0.1           | Post attention dropout probability.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Meg    | hidden-dropout                            | 0.1           | Dropout probability for hidden state transformer.                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Meg    | weight-decay                              | 0.01          | Weight decay coefficient for L2 regularization.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |

### Activation Checkpointing Settings:

```yaml
   "checkpoint-activations": true,
   "checkpoint-num-layers": 1,
   "partition-activations": true,
   "synchronize-each-layer": true,
```

Checkpointing works by trading compute for memory. Rather than storing all intermediate activations of the entire computation graph for computing backward, the checkpointed part does not save intermediate activations, and instead recomputes them in backward pass.
All options for configuring activation checkpointing are below:

| Origin | Parameter name                            | Default value | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|--------|-------------------------------------------|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Meg    | checkpoint-activations                    | false         | Checkpoint activation to allow for training with larger models, sequences, and batch sizes.                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Meg    | checkpoint-num-layers                     | 1             | Chunk size (number of layers) for checkpointing.                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Meg    | distribute-checkpointed-activations       | false         | If set, distribute checkpointed activations across model parallel group.                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Meg    | deepspeed-activation-checkpointing        | false         | Uses activation checkpointing from deepspeed                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Meg    | contiguous-checkpointing                  | false         | Contiguous memory checkpointing for activations.                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Meg    | checkpoint-in-cpu                         | false         | Move the activation checkpoints to CPU.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Meg    | synchronize-each-layer                    | false         | does a synchronize at the beginning and end of each checkpointed layer.                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Meg    | profile-backward                          | false         | Enables backward pass profiling for checkpointed layers.                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Meg    | partition-activations                     | false         | Partition Activations across GPUs before checkpointing.                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |

### Mixed Precision Training Settings:
gpt-neox's mixed precision training is configured identically to DeepSpeed's, please see [their documentation](https://www.deepspeed.ai/docs/config-json/#fp16-training-options) for more information.
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

To train in fp32, simply set `fp16["enabled"]` to `false`.

### Deepspeed Launcher Options

| Origin | Parameter name                            | Default value | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|--------|-------------------------------------------|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DSL    | hostfile                                  | /job/hostfile | Hostfile path (in MPI style) that defines the resource pool available to the job (e.g., `worker-0 slots=4`)                                                                                                                                                                                                                                                                                                                                                                                                                    |
| DSL    | include                                   |               | Specify hardware resources to use during execution. String format is `NODE_SPEC[@NODE_SPEC ...]` where `NODE_SPEC=NAME[:SLOT[,SLOT ...]]`. If `:SLOT` is omitted, include all slots on that host. Example: `"worker-0@worker-1:0,2"` will use all slots. on `worker-0` and slots `[0, 2]` on `worker-1`.                                                                                                                                                                                                                       |
| DSL    | exclude                                   |               | Specify hardware resources to NOT use during execution. Same format as `include`.                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| DSL    | num_nodes                                 | -1            | Total number of worker nodes to run on, this will use the top N hosts from the given hostfile. `-1` will use all.                                                                                                                                                                                                                                                                                                                                                                                                              |
| DSL    | num_gpus                                  | -1            | Max number of GPUs to use on each node, will use [0:N) GPU ids on each node. `-1` will use all.                                                                                                                                                                                                                                                                                                                                                                                                                                |
| DSL    | master_port                               | 29500         | Port used by PyTorch distributed for communication during training.                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| DSL    | master_addr                               |               | IP address of node 0, will be inferred via 'hostname -I' if not specified.                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| DSL    | launcher                                  | pdsh          | Launcher backend for multi-node training. Options currently include PDSH, OpenMPI, MVAPICH.                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| DSL    | detect_nvlink_pairs                       | false         | If true, autodetects nvlink pairs and remaps cuda visible devices to place them next to each other. This is an Eleuther addition to deepspeed, and should speed up model parallel training on setups with nvlink pairs when mp=2.                                                                                                                                                                                                                                                                                                                                                                                                                                    |


### Other Configuration Options

And finally, a few leftover options that don't fit into any particular category.

| Origin | Parameter name                            | Default value | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|--------|-------------------------------------------|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Meg    | distributed-backend                       | nccl          | Which backend to use for distributed training.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Meg    | local_rank                                |               | local rank passed from distributed launcher.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Meg    | lazy-mpu-init                             |               | If set to True, initialize_megatron() skips DDP initialization and returns function to complete it instead. Also turns on `use-cpu-initialization` flag. This is for external DDP manager.                                                                                                                                                                                                                                                                                                                                     |
| Meg    | short-seq-prob                            | 0.1           | Probability of producing a short sequence.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Meg    | reset-position-ids                        | false         | Reset posistion ids after end-of-document token.                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Meg    | reset-attention-mask                      | false         | Reset self attention maske after end-of-document token.                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Meg    | eod-mask-loss                             | false         | Mask loss for the end of document tokens.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Meg    | adlr-autoresume                           | false         | Enable auto-resume on adlr cluster.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| Meg    | adlr-autoresume-interval                  | 1000          | Intervals over which check for auto-resume termination signal                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Meg    | seed                                      | 1234          | Random seed used for python, numpy, pytorch, and cuda.                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| NeoX   | deepspeed_mpi                             | false         | Run via MPI, this will attempt to discover the necessary variables to initialize torch distributed from the MPI environment                                                                                                                                                                                                                                                                                                                                                                                                    |
| DSC    | prescale_gradients                        | false         | Scale gradients before doing allreduce.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| DSC    | gradient_predivide_factor                 | 1.0           | Before gradient averaging predivide gradients by a specified factor, can sometimes help with fp16 stability when scaling to large numbers of GPUs                                                                                                                                                                                                                                                                                                                                                                              |
| DSC    | sparse_gradients                          | false         | Enable sparse compression of torch.nn.Embedding gradients.                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| DSC    | amp                                       |               | [Dictionary as described in Deepspeed documentation.](https://www.deepspeed.ai/docs/config-json/#automatic-mixed-precision-amp-training-options)                                                                                                                                                                                                                                                                                                                                                                               |
| DSC    | flops_profiler                            |               | [Dictionary as described in Deepspeed documentation.](https://www.deepspeed.ai/docs/config-json/#flops-profiler)                                                                                                                                                                                                                                                                                                                                                                                                               |
