Arguments for gpt-neox. All of the following can be specified in your .yml config file(s):


## NeoXArgsLRScheduler

LR Scheduler Arguments



- **lr_decay_style**: typing.Literal['constant', 'linear', 'cosine', 'exponential']

    Default = linear

    Learning rate decay function. Choose from 'constant', 'linear', 'cosine', 'exponential'.



- **lr_decay_iters**: <class 'int'>

    Default = None

    Number of iterations to decay learning rate over, If None defaults to --train-iters



- **min_lr**: <class 'float'>

    Default = 0.0

    Minimum value for learning rate. The scheduler clips values below this threshold.



- **warmup**: <class 'float'>

    Default = 0.01

    Percentage of total iterations to warmup on (.01 = 1 percent of all training iters).



- **override_lr_scheduler**: <class 'bool'>

    Default = False

    Reset the values of the scheduler (learning rate,warmup iterations, minimum learning rate, maximum number of iterations, and decay style from input arguments and ignore values from checkpoints. Note that all the above values will be reset.



- **use_checkpoint_lr_scheduler**: <class 'bool'>

    Default = False

    Use checkpoint to set the values of the scheduler (learning rate, warmup iterations, minimum learning rate, maximum number of iterations, and decay style from checkpoint and ignore input arguments.



## NeoXArgsLogging

Logging Arguments



- **use_wandb**: <class 'bool'>

    Default = None

    Flag indicating if wandb is to be used.



- **wandb_group**: <class 'str'>

    Default = None

    Weights and Biases group name - used to group together "runs".



- **wandb_team**: <class 'str'>

    Default = None

    Team name for Weights and Biases.



- **wandb_project**: <class 'str'>

    Default = neox

    wandb project name



- **wandb_host**: <class 'str'>

    Default = https://api.wandb.ai

    url of the wandb host



- **wandb_init_all_ranks**: <class 'bool'>

    Default = False

    Initialize wandb on all ranks.



- **git_hash**: <class 'str'>

    Default = b5024e3

    current git hash of repository



- **log_dir**: <class 'str'>

    Default = None

    Directory to save logs to.



- **tensorboard_dir**: <class 'str'>

    Default = None

    Write TensorBoard logs to this directory.



- **log_interval**: <class 'int'>

    Default = 100

    Interval between logging.



- **log_grad_pct_zeros**: <class 'bool'>

    Default = False

    Log the percentage of zeros for the gradient of each parameter to wandb / tensorboard (useful for debugging). Needs wandb_init_all_ranks set to True if using pipeline parallelism to log all ranks.



- **log_param_norm**: <class 'bool'>

    Default = False

    Log the frob norm of the parameters to wandb / tensorboard (useful for debugging). Needs wandb_init_all_ranks set to True if using pipeline parallelism to log all ranks.



- **log_grad_norm**: <class 'bool'>

    Default = False

    Log the frob norm of the gradients to wandb / tensorboard (useful for debugging).
    (N.B - this will only work with pp = 0 for now, as we don't have access to the gradients of the model because
    deepspeed.)



- **log_optimizer_states**: <class 'bool'>

    Default = False

    Log the frob norm of the optimizer states to wandb / tensorboard (useful for debugging).



- **log_gradient_noise_scale**: <class 'bool'>

    Default = False

    Whether to log the gradient noise scale when training (cf. https://arxiv.org/abs/1812.06162 for explanation)



- **gradient_noise_scale_n_batches**: <class 'int'>

    Default = 5

    Number of batches to accumulate gradients for in the gradient noise scale logger.



- **gradient_noise_scale_cpu_offload**: <class 'bool'>

    Default = False

    Whether to offload the buffered gradients to cpu when measuring gradient noise scale.



- **memory_profiling**: <class 'bool'>

    Default = False

    Whether to take a memory snapshot of the model. Useful for debugging memory issues.



- **memory_profiling_path**: <class 'str'>

    Default = None

    Path to save memory snapshot to.



- **profile**: <class 'bool'>

    Default = False

    Enable nsys profiling. When using this option,
    nsys options should be specified in commandline.
    An example nsys commandline is
    ```
    nsys profile -s none -t nvtx,cuda -o <path/to/output_file>
    --force-overwrite true
    --capture-range=cudaProfilerApi
    --capture-range-end=stop
    ```



- **profile_step_start**: <class 'int'>

    Default = 10

    Step to start profiling at.



- **profile_step_stop**: <class 'int'>

    Default = 12

    Step to stop profiling at.



## NeoXArgsModel

Model Arguments



- **precision**: typing.Literal['fp16', 'fp32', 'bfloat16']

    Default = None

    description of the used precision, either one of fp16 or fp32 (and in the future bf16).



- **num_layers**: <class 'int'>

    Default = None

    Number of transformer layers.



- **hidden_size**: <class 'int'>

    Default = None

    Transformer hidden size.



- **intermediate_size**: <class 'int'>

    Default = None

    Transformer intermediate size. Currently only used for "mlp_type": "llama".

    If not passed, will be set to a reasonable default.



- **num_attention_heads**: <class 'int'>

    Default = None

    Number of transformer attention heads.

    If num_kv_heads is set, will control only number of query heads.



- **num_kv_heads**: <class 'int'>

    Default = None

    Number of transformer key/value attention heads.

    If set to None or the same value as num_attention_heads, will perform multi-head attention (MHA).
    If set to < num_attention_heads but > 1, will perform grouped-query attention (GQA) (https://arxiv.org/pdf/2305.13245.pdf)
    If set to 1, will perform multi-query attention.

    Must be < num_attention_heads and divide num_attention_heads evenly.



- **seq_length**: <class 'int'>

    Default = None

    Maximum sequence length to process.



- **sliding_window_width**: <class 'int'>

    Default = None

    Width of the attention sliding window. Only supported with Flash Attention 2.



- **max_position_embeddings**: <class 'int'>

    Default = None

    Maximum number of position embeddings to use. This is the size of position embedding.



- **norm**: typing.Literal['layernorm', 'rmsnorm', 'scalenorm']

    Default = layernorm

    Normalization layer to use. Choose from "layernorm", "rmsnorm", "scalenorm".



- **layernorm_fusion**: <class 'bool'>

    Default = False

    Use fused layer norm kernel (if `norm` is `layernorm`).



- **use_qk_layernorm**: <class 'bool'>

    Default = False

    Use QK Normalization



- **layernorm_epsilon**: <class 'float'>

    Default = 1e-05

    Layer norm epsilon.



- **rms_norm_epsilon**: <class 'float'>

    Default = 1e-08

    Root mean squared norm epsilon



- **scalenorm_epsilon**: <class 'float'>

    Default = 1e-08

    Scalenorm epsilon



- **pos_emb**: typing.Literal['learned', 'rotary', 'sinusoidal', 'rpe', 'alibi', 'none']

    Default = learned

    Type of positional embedding to use - choose from 'learned', 'rotary', 'sinusoidal', 'rpe', 'none'



- **rpe_num_buckets**: <class 'int'>

    Default = 32

    T5 relative positional encoding number of buckets, default 32.



- **rpe_max_distance**: <class 'int'>

    Default = 128

    T5 relative positional encoding max distance, default 128.



- **opt_pos_emb_offset**: <class 'int'>

    Default = 0

    Learned position embedding offset (only used by OPT, where it should be set to 2).



- **no_weight_tying**: <class 'bool'>

    Default = False

    Disables weight tying between embedding weights and final Linear layer



- **attention_config**: <class 'list'>

    Default = None

    Attention configuration for gpt-neox

    The first item in the list specifies the attention type(s), and should be a list of strings. The second item
    specifies the number of times to repeat those attention types in the full list.

    attention type choices:  [global, local, sparse_fixed, sparse_variable, bslongformer, bigbird, "gmlp", "amlp", "flash"]

    So a 12 layer network with only global attention could be specified like:
        [[[`global`], 12]]

    or a 12 layer network with alternating global / local like:
        [[[`global`, `local`], 6]]

    If none is specified, this defaults to
        [[[`global`], n_layers]]



- **sparsity_config**: <class 'dict'>

    Default = None

    Sparsity configuration dict as defined in https://www.deepspeed.ai/docs/config-json/#sparse-attention

    Note that since neox is autoregressive, attention is always "unidirectional" and `horizontal_global_attention` is
    always false.

    The main difference between our sparsity config and deepspeed's is that `mode` is ignored - since it is instead
    specified in attention_config defining each layer.

    An example config is given below:
          "sparse_attention": {
            "block": 16,
            "different_layout_per_head": true,
            "num_local_blocks": 4,
            "num_global_blocks": 1,
            "num_different_global_patterns": 4,
            "num_random_blocks": 0,
            "local_window_blocks": [4],
            "global_block_indices": [0],
            "global_block_end_indices": None,
            "num_sliding_window_blocks": 3
          }



- **num_unique_layers**: <class 'int'>

    Default = None

    Number of unique transformer layers. num-layers should be divisible by this value. Currently only has an effect when pipe_parallel_size=0.



- **param_sharing_style**: <class 'str'>

    Default = grouped

    Ordering of the shared parameters. For example, for a num-layers=4 and --num-unique-layers=2, we will have the following ordering for two unique layers 1 and 2-: grouped: [1, 2, 1, 2] and spaced: [1, 1, 2, 2].



- **make_vocab_size_divisible_by**: <class 'int'>

    Default = 128

    Pad the vocab size to be divisible by this value. This is added for computational efficiency reasons.



- **activation**: typing.Literal['gelu', 'geglu', 'relu', 'softsign', 'swish', 'mish', 'silu']

    Default = gelu

    Activation function to use - choose from ["gelu", "geglu", "relu", "softsign", "swish", "mish", "silu"]



- **scaled_upper_triang_masked_softmax_fusion**: <class 'bool'>

    Default = False

    Enable fusion of query_key_value_scaling time (upper diagonal) masking and softmax.



- **scaled_masked_softmax_fusion**: <class 'bool'>

    Default = False

    Enable fusion of query_key_value_scaling general masking and softmax.



- **bias_gelu_fusion**: <class 'bool'>

    Default = False

    Enable bias and gelu fusion.



- **bias_dropout_fusion**: <class 'bool'>

    Default = False

    Enable bias and dropout fusion.



- **rope_fusion**: <class 'bool'>

    Default = False

    Enable rotary embedding fusion.



- **fp16_lm_cross_entropy**: <class 'bool'>

    Default = False

    Move the cross entropy unreduced loss calculation for lm head to fp16.



- **init_method_std**: <class 'float'>

    Default = 0.02

    Standard deviation of the zero mean normal distribution used for weight initialization.



- **apply_query_key_layer_scaling**: <class 'bool'>

    Default = False

    Scale Q * K^T by 1 / layer-number. If this flag is set, then it will automatically set attention-softmax-in-fp32 to true



- **use_cpu_initialization**: <class 'bool'>

    Default = False

    If set, affine parallel weights initialization uses CPU



- **attention_softmax_in_fp32**: <class 'bool'>

    Default = False

    Run attention masking and softmax in fp32.



- **rotary_pct**: <class 'float'>

    Default = 1.0

    pct of hidden dims to apply rotary positional embedding to



- **rotary_emb_base**: <class 'int'>

    Default = 10000

    Base for rotary positional embedding



- **init_method**: typing.Literal['normal', 'scaled_normal', 'orthogonal', 'scaled_orthogonal', 'xavier_uniform', 'xavier_normal', 'wang_init', 'small_init']

    Default = normal

    Init function used on all layers except ff residual outputs - choose from
    ["normal", "scaled_normal", "orthogonal", "scaled_orthogonal", "xavier_uniform", "xavier_normal", "wang_init", "small_init"]



- **output_layer_init_method**: typing.Literal['normal', 'scaled_normal', 'orthogonal', 'scaled_orthogonal', 'xavier_uniform', 'xavier_normal', 'wang_init', 'small_init']

    Default = scaled_normal

    Init function used for ff residual outputs - choose from
    ["normal", "scaled_normal", "orthogonal", "scaled_orthogonal", "xavier_uniform", "xavier_normal", "wang_init", "small_init"]



- **gmlp_attn_dim**: <class 'int'>

    Default = 64

    the dimension of the single head self attention in gmlp model (not used in gpt models).
    If None - gmlp model doesn't use attention.



- **gpt_j_residual**: <class 'bool'>

    Default = False

    If false, we use the conventional residual path:
      x = x + attn(ln1(x))
      x = x + mlp(ln2(x))
    Otherwise, we use the residual path from GPT-J, which offers a slight speedup:
      x = ln(x)
      x = x + attn(x) + mlp(x)



- **gpt_j_tied**: <class 'bool'>

    Default = False

    If false, we use
      x = x + attn(ln1(x)) + mlp(ln2(x))
    Otherwise, we tie the layer norms
      y = ln(x)
      x = x + attn(y) + mlp(y)



- **use_bias_in_norms**: <class 'bool'>

    Default = True

    If false, norms (e.g. LayerNorm) will not have bias terms



- **use_bias_in_attn_linear**: <class 'bool'>

    Default = True

    If false, attn_linear (e.g. QKVO) will not have bias terms



- **mlp_type**: <class 'str'>

    Default = regular

    Types:
        regular: Megatron implementation
        llama: LLaMA MLP (SiLU-gated MLP)



- **soft_prompt_tuning**: <class 'dict'>

    Default = None

    Dictionary configuring the soft prompt tuning parameters.
    If enabled, will train *only* the soft prompt, and freezes the rest of the model.
    parameters in the dict are:
        'enabled': bool = True # enables soft prompting
        'num_tokens': int = 10 # length of the soft prompt in tokens
        'init_string': str = '' # if provided, initialize the soft prompt with the word embeddings of this string
        'init_range': float = 0.5 # if no init string is provided, initialize the soft prompt with a uniform distribution between -init_range and init_rang



- **output_layer_parallelism**: typing.Literal['column']

    Default = column

    Parameter controlling whether the output layer is parallelized over the hidden dim (row) or the vocab dim (column)



## NeoXArgsOptimizer

Optimizer Arguments



- **optimizer_type**: typing.Literal['adam', 'onebitadam', 'cpu_adam', 'cpu_torch_adam', 'sm3', 'madgrad_wd', 'sgd', 'lion']

    Default = adam

    Type of optimizer to use. Choose from ['adam', 'onebitadam', 'cpu_adam', 'cpu_torch_adam', 'sm3', 'madgrad_wd', 'sgd', 'lion']
    NOTE: sgd will use MuSGD from Mup. Mup must be enabled for this optimizer.



- **use_bnb_optimizer**: <class 'bool'>

    Default = False

    Whether to enable the bitsandbytes optimizers



- **zero_stage**: typing.Union[int, typing.List[int], typing.Literal['all']]

    Default = None

    Zero Optimizer stage



- **zero_reduce_scatter**: <class 'bool'>

    Default = None

    Zero: Uses reduce or reduce scatter instead of allreduce to average gradients



- **zero_contiguous_gradients**: <class 'bool'>

    Default = None

    Zero: Copies the gradients to a contiguous buffer as they are produced. Avoids memory fragmentation during backward pass. Only useful when running very large models.



- **zero_reduce_bucket_size**: <class 'int'>

    Default = None

    Zero: Number of elements reduced/allreduced at a time. Limits the memory required for the allgather for large model sizes



- **zero_allgather_bucket_size**: <class 'int'>

    Default = None

    Zero: Number of elements allgathered at a time. Limits the memory required for the allgather for large model sizes



- **lr**: <class 'float'>

    Default = None

    Max Learning rate during training



## NeoXArgsOther

Misc. Arguments



- **distributed_backend**: <class 'str'>

    Default = nccl

    Which backend to use for distributed training.



- **local_rank**: <class 'int'>

    Default = None

    local rank passed from distributed launcher.



- **rank**: <class 'int'>

    Default = None

    global rank of process being run (passed in via distributed launcher)



- **lazy_mpu_init**: <class 'bool'>

    Default = False

    If set to True, initialize_megatron() skips DDP initialization and returns function to complete it instead. Also turns on use-cpu-initialization flag. This is for external DDP manager.



- **short_seq_prob**: <class 'float'>

    Default = 0.1

    Probability of producing a short sequence.



- **eod_mask_loss**: <class 'bool'>

    Default = False

    Mask loss for the end of document tokens.



- **adlr_autoresume**: <class 'bool'>

    Default = False

    Enable auto-resume on adlr cluster.



- **adlr_autoresume_interval**: <class 'int'>

    Default = 1000

    Intervals over which check for auto-resume termination signal



- **seed**: <class 'int'>

    Default = 1234

    Random seed used for python, numpy, pytorch, and cuda.



- **onnx_safe**: <class 'bool'>

    Default = False

    Use workarounds for known problems with Torch ONNX exporter



- **deepscale**: <class 'bool'>

    Default = False

    (Deprecated) enable DeepSpeed (helper flag for user code, no impact on DeepSpeed backend)'



- **deepscale_config**: <class 'str'>

    Default = None

    (Deprecated) deepscale json configuration file.



- **deepspeed_mpi**: <class 'bool'>

    Default = False

    Run via MPI, this will attempt to discover the necessary variables to initialize torch distributed from the MPI environment



- **deepspeed_slurm**: <class 'bool'>

    Default = False

    Run via SLURM, this will attempt to discover the necessary variables to initialize torch distributed from the SLURM environment



- **user_script**: <class 'str'>

    Default = None

    user script to be run



- **iteration**: <class 'int'>

    Default = None

    Set during training



- **do_train**: <class 'bool'>

    Default = None

    Set during training



- **do_valid**: <class 'bool'>

    Default = None

    Set during training



- **do_test**: <class 'bool'>

    Default = None

    Set during training



- **save_iters**: <class 'list'>

    Default = None

    Set during training



- **global_num_gpus**: <class 'int'>

    Default = None

    Set during launching



## NeoXArgsParallelism

Parallelism Arguments



- **pipe_parallel_size**: <class 'int'>

    Default = 0

    Number of pipeline parallel stages. Disable with 0.



- **model_parallel_size**: <class 'int'>

    Default = 1

    Size of the model parallelism.



- **pipe_partition_method**: <class 'str'>

    Default = type:transformer|mlp

    method used to distribute model layers across pipeline stages. Choose from "parameters", which balances the number
    of parameters on each pipeline stage, "uniform", which naively balances the number of layers per stage, or
    "type:[regex]", which balances layers whose class names match [regex]



- **world_size**: <class 'int'>

    Default = None

    Total world size (i.e number of gpus in cluster). Configured post-launch using distributed launcher



- **is_pipe_parallel**: <class 'bool'>

    Default = False

    flag to determine whether pipeline parallelism is on - shouldn't be set by user, is automatically determined
    according to pipeline parallel size.



## NeoXArgsTemplate

NeoXArgsTemplate()



## NeoXArgsTextgen

Text Generation arguments



- **text_gen_type**: <class 'str'>

    Default = None

    How to generate text/sample the model.
    Options: `unconditional`, `input-file`, `interactive`



- **temperature**: <class 'float'>

    Default = 0.0

    exponential scaling output distribution ("higher == more risk")



- **top_p**: <class 'float'>

    Default = 0.0

    Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.



- **top_k**: <class 'int'>

    Default = 0

    integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.



- **return_logits**: <class 'bool'>

    Default = False

    Boolean for whether to return the logits for generated tokens



- **maximum_tokens**: <class 'int'>

    Default = 64

    maximum number of tokens to be generated



- **prompt_end**: <class 'str'>

    Default = 


    a single prompt's end. Defaults to newline



- **sample_input_file**: <class 'str'>

    Default = None

    Get input from file instead of interactive mode, each line is an input.



- **sample_output_file**: <class 'str'>

    Default = samples.txt

    Output file



- **num_samples**: <class 'int'>

    Default = 1

    Number of samples to generate unconditionally, defaults to 1 and interactive conditional sampling



- **recompute**: <class 'bool'>

    Default = False

    During generation recompute all attention instead of using previously computed keys/values.
    Should be set to true for sparse attention models



- **eval_results_prefix**: <class 'str'>

    Default = 

    prefix to which to save evaluation results - final fp will be {eval_results_prefix}_eval_results_yy-mm-dd-HH-MM.json



- **eval_tasks**: <class 'list'>

    Default = None

    Tasks to evaluate on using lm_eval_harness

    NOTE: Requires internet connection



## NeoXArgsTokenizer

Tokenizer Arguments



- **tokenizer_type**: typing.Literal['GPT2BPETokenizer', 'HFTokenizer', 'HFGPT2Tokenizer', 'SPMTokenizer', 'CharLevelTokenizer', 'TiktokenTokenizer']

    Default = GPT2BPETokenizer

    Type of tokenizer to use - should be one of ["GPT2BPETokenizer", "HFTokenizer", "HFGPT2Tokenizer", "SPMTokenizer", "CharLevelTokenizer", "TiktokenTokenizer"]



- **padded_vocab_size**: <class 'int'>

    Default = None

    Total (padded) vocabulary size of tokenizer. Configured after launching of training,
    as it's dependent on the parallelism size.



## NeoXArgsTraining

Training Arguments



- **data_path**: <class 'str'>

    Default = None

    Path to combined dataset to split.



- **use_shared_fs**: <class 'bool'>

    Default = True

    Whether to use a shared filesystem for data loading. If False, local rank 0 on all nodes will preprocess the data,
    otherwise only global rank 0 will preprocess the data. This is implemented in megatron/data/gpt2_dataset.py::_build_index_mappings.



- **train_data_paths**: <class 'list'>

    Default = None

    List of paths to train datasets.



- **label_data_paths**: <class 'list'>

    Default = None

    List of paths to label datasets (not shifted by 1 yet!).



- **test_data_paths**: <class 'list'>

    Default = None

    List of paths to test datasets.



- **valid_data_paths**: <class 'list'>

    Default = None

    List of paths to validation datasets.



- **train_data_weights**: <class 'list'>

    Default = None

    List of 'weights' that decide how often to sample from each training dataset when blending datasets. If None, defaults to equal weighting.
    Should be a list the same length as `train_data_paths`



- **valid_data_weights**: <class 'list'>

    Default = None

    List of 'weights' that decide how often to sample from each validation dataset when blending datasets. If None, defaults to equal weighting.
    Should be a list the same length as `valid_data_paths`



- **test_data_weights**: <class 'list'>

    Default = None

    List of 'weights' that decide how often to sample from each test dataset when blending datasets. If None, defaults to equal weighting.
    Should be a list the same length as `test_data_paths`



- **weight_by_num_documents**: <class 'bool'>

    Default = False

    If True, Builds dataset weights from a multinomial distribution over groups of data according to the number of
    documents in each group.

    WARNING: setting this to True will override any user provided weights

    We sample from a group according to the probability p(L) ∝ |L| ** α,
    where p(L) is the probability of sampling from a given group,
          |L| is the number of examples in that datapoint,
          and α is a coefficient that acts to upsample data from underrepresented groups

    Hence α (`alpha`) allows us to control how much to 'boost' the probability of training on low-resource groups.

    See https://arxiv.org/abs/1911.02116 for more details



- **weighted_sampler_alpha**: <class 'float'>

    Default = 1.0

    Alpha value for `weight_by_num_documents`. Only has an effect if `weight_by_num_documents` = True.

    when alpha = 1, the probability of sampling from a given group = n_samples / total_samples
    as alpha -> 0, the probability of sampling from all groups becomes equal, and number of documents has no effect
    as alpha -> inf, the probability of sampling from the groups with *the most samples* -> 1



- **data_impl**: typing.Literal['infer', 'mmap', 'cached']

    Default = infer

    Implementation of indexed datasets, can be one of "infer", "cached", or "mmap"



- **mmap_warmup**: <class 'bool'>

    Default = False

    Warm up mmap files.



- **save**: <class 'str'>

    Default = None

    Output directory to save checkpoints to.



- **s3_path**: <class 'str'>

    Default = None

    Path to s3 bucket for saving checkpoints.



- **s3_chunk_size**: <class 'int'>

    Default = 104857600

    The number of bytes in each file chunk when uploading to s3. Defaults to 100MiB.



- **config_files**: <class 'dict'>

    Default = None

    Store of original config files mapping config filename to file contents



- **load**: <class 'str'>

    Default = None

    Directory containing a model checkpoint.



- **checkpoint_validation_with_forward_pass**: <class 'bool'>

    Default = False

    save input and output of a forward pass with the checkpoint and validate after load



- **checkpoint_scale**: typing.Literal['linear', 'log']

    Default = linear

    How step at which checkpoints are saved should scale. "linear" implies 1 checkpoint will be saved at every multiple of `checkpoint-factor`,
    while "log" implies that the number of steps between each checkpoint will be multiplied by `checkpoint-factor` at each step, starting from step 1.



- **checkpoint_factor**: <class 'int'>

    Default = None

    Acts as a multiplier on either the "log" or "linear" checkpoint spacing.

    With `checkpoint-scale="linear"`, `checkpoint-factor=20`, and `train-iters=100`, checkpoints will be saved at
    steps [20, 40, 60, 80, 100].

    With `checkpoint-scale="log"`, `checkpoint-factor=2`, and `train-iters=100`, checkpoints will be saved at
    steps [1, 2, 4, 8, 16, 32, 64, 100].

    Note that the last checkpoint step is always saved.



- **extra_save_iters**: <class 'list'>

    Default = None

    Additional iterations when a checkpoint should be saved.
    Must be a list of ints or `None`.



- **no_save_optim**: <class 'bool'>

    Default = False

    Do not save current optimizer.



- **no_save_rng**: <class 'bool'>

    Default = False

    Do not save current rng state.



- **no_load_optim**: <class 'bool'>

    Default = False

    Do not load optimizer when loading checkpoint.



- **no_load_rng**: <class 'bool'>

    Default = False

    Do not load rng state when loading checkpoint.



- **finetune**: <class 'bool'>

    Default = False

    Load model for finetuning. Do not load optimizer or rng state from checkpoint and set iteration to 0. Assumed when loading a release checkpoint.



- **batch_size**: <class 'int'>

    Default = None

    training microbatch size per gpu



- **train_iters**: <class 'int'>

    Default = None

    Number of iterations to run for training.



- **eval_iters**: <class 'int'>

    Default = 100

    Number of iterations to run for evaluation validation/test for.



- **keep_last_n_checkpoints**: <class 'int'>

    Default = None

    Number of last checkpoints to keep



- **eval_interval**: <class 'int'>

    Default = 1000

    Interval between running evaluation on validation set.



- **split**: <class 'str'>

    Default = 969, 30, 1

    Comma_separated list of proportions for training, validation, and test split. For example the split 90,5,5 will use 90% of data for training, 5% for validation and 5% for test.



- **vocab_file**: <class 'str'>

    Default = None

    Path to the vocab file.



- **merge_file**: <class 'str'>

    Default = None

    Path to the BPE merge file.



- **num_workers**: <class 'int'>

    Default = 2

    Dataloader number of workers.



- **exit_interval**: <class 'int'>

    Default = None

    Exit the program after the iteration is divisible by this value.



- **attention_dropout**: <class 'float'>

    Default = 0.0

    Post attention dropout probability.



- **hidden_dropout**: <class 'float'>

    Default = 0.0

    Dropout probability for hidden state transformer.



- **weight_decay**: <class 'float'>

    Default = 0.1

    Weight decay coefficient for L2 regularization.



- **checkpoint_activations**: <class 'bool'>

    Default = False

    Checkpoint activation to allow for training with larger models, sequences, and batch sizes.



- **checkpoint_num_layers**: <class 'int'>

    Default = 1

    Chunk size (number of layers) for checkpointing.



- **deepspeed_activation_checkpointing**: <class 'bool'>

    Default = True

    DEPRECATED - TODO: remove
    Uses activation checkpointing from deepspeed



- **contiguous_checkpointing**: <class 'bool'>

    Default = False

    Contiguous memory checkpointing for activations.



- **checkpoint_in_cpu**: <class 'bool'>

    Default = False

    Move the activation checkpoints to CPU.



- **synchronize_each_layer**: <class 'bool'>

    Default = False

    does a synchronize at the beginning and end of each checkpointed layer.



- **profile_backward**: <class 'bool'>

    Default = False

    Enables backward pass profiling for checkpointed layers.



- **partition_activations**: <class 'bool'>

    Default = False

    Partition Activations across GPUs before checkpointing.



- **clip_grad**: <class 'float'>

    Default = 1.0

    Gradient clipping based on global L2 norm.



- **hysteresis**: <class 'int'>

    Default = 2

    hysteresis for dynamic loss scaling



- **dynamic_loss_scale**: <class 'bool'>

    Default = None

    flag indicating whether dynamic loss scale is used



- **loss_scale**: <class 'float'>

    Default = None

    Static loss scaling, positive power of 2
    values can improve fp16 convergence. If None, dynamic loss scaling is used.



- **loss_scale_window**: <class 'float'>

    Default = 1000.0

    Window over which to raise/lower dynamic scale.



- **min_scale**: <class 'float'>

    Default = 1.0

    Minimum loss scale for dynamic loss scale.



- **char_level_ppl**: <class 'bool'>

    Default = False

    Whether to calculate character level perplexity as well as token level perplexity. (may incur a time cost)



- **use_mup**: <class 'bool'>

    Default = False

    Whether to use Microsoft's Mup https://github.com/microsoft/mup



- **coord_check**: <class 'bool'>

    Default = False

    Whether to generate a "coord check" plot to verify mup's implementation in neox



- **save_base_shapes**: <class 'bool'>

    Default = False

    Whether to save base shapes for mup. This will save the shapes to the path specified in base-shapes-file.



- **base_shapes_file**: <class 'str'>

    Default = None

    Path to the base shapes to save to/load from



- **mup_init_scale**: <class 'float'>

    Default = 1.0

    Initialization scale: All the parameters are multiplied by this value



- **mup_attn_temp**: <class 'float'>

    Default = 1.0

    Attention temperature: Reciprocal of the multiplier applied to the input to attention softmax



- **mup_output_temp**: <class 'float'>

    Default = 1.0

    Output temperature: Reciprocal of the multiplier applied to the input to softmax that
    produces the distribution over output tokens.



- **mup_embedding_mult**: <class 'float'>

    Default = 1.0

    Scalar by which we multiply the output of the embedding layer



- **mup_rp_embedding_mult**: <class 'float'>

    Default = 1.0

    Scalar by which we multiply vectors representing relative position



- **mup_width_scale**: <class 'int'>

    Default = 2

    What to scale width by when creating the delta model for mup



## NeoXArgsDeepspeedConfig

Args for deepspeed config
    Every argument included here will be included in deepspeed config json
    As of Mar 8 2023, up to date compared to https://www.deepspeed.ai/docs/config-json/



- **deepspeed**: <class 'bool'>

    Default = True

    boolean flag to enable DeepSpeed (Always True)



- **train_batch_size**: <class 'int'>

    Default = None

    The effective training batch size. This is the amount of data samples that leads to one step of model update. train_batch_size is aggregated by the batch size that a single GPU processes in one forward/backward pass (a.k.a., train_step_batch_size), the gradient accumulation steps (a.k.a., gradient_accumulation_steps), and the number of GPUs.



- **train_micro_batch_size_per_gpu**: <class 'int'>

    Default = None

    Batch size to be processed by one GPU in one step (without gradient accumulation). When specified, gradient_accumulation_steps is automatically calculated using train_batch_size and number of GPUs. Should not be concurrently specified with gradient_accumulation_steps in the configuration JSON.



- **gradient_accumulation_steps**: <class 'int'>

    Default = 1

    Number of training steps to accumulate gradients before averaging and applying them. This feature is sometimes useful to improve scalability since it results in less frequent communication of gradients between steps. Another impact of this feature is the ability to train with larger batch sizes per GPU. When specified, train_step_batch_size is automatically calculated using train_batch_size and number of GPUs. Should not be concurrently specified with train_step_batch_size in the configuration JSON.



- **optimizer**: <class 'dict'>

    Default = None

    dict containing the keys type and params

    type: The optimizer name. DeepSpeed natively supports Adam, AdamW, OneBitAdam, Lamb, and OneBitLamb optimizers (See here for details) and will import other optimizers from torch.

    params: Dictionary of parameters to instantiate optimizer. The parameter names must match the optimizer constructor signature (e.g., for Adam).



- **scheduler**: <class 'dict'>

    Default = None

    dict containing the keys type and params

    type: The scheduler name. See here (https://deepspeed.readthedocs.io/en/latest/schedulers.html) for list of support schedulers.

    params: Dictionary of parameters to instantiate scheduler. The parameter names should match scheduler constructor signature.



- **fp32_allreduce**: <class 'bool'>

    Default = False

    During gradient averaging perform allreduce with 32 bit values



- **prescale_gradients**: <class 'bool'>

    Default = False

    Scale gradients before doing allreduce



- **gradient_predivide_factor**: <class 'float'>

    Default = 1.0

    Before gradient averaging predivide gradients by a specified factor, can sometimes help with fp16 stability when scaling to large numbers of GPUs



- **sparse_gradients**: <class 'bool'>

    Default = False

    Enable sparse compression of torch.nn.Embedding gradients.



- **fp16**: <class 'dict'>

    Default = None

    Configuration for using mixed precision/FP16 training that leverages NVIDIA’s Apex package.

    Dictionary options as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#fp16-training-options



- **bf16**: <class 'dict'>

    Default = None

    Configuration for using bfloat16 floating-point format as an alternative to FP16. BFLOAT16 requires hardware support (e.g., NVIDIA A100). Dictionary options as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#bfloat16-training-options



- **amp**: <class 'dict'>

    Default = None

    Configuration for using automatic mixed precision (AMP) training that leverages NVIDIA’s Apex AMP package.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#automatic-mixed-precision-amp-training-options



- **gradient_clipping**: <class 'float'>

    Default = 1.0

    Enable gradient clipping with provided value



- **zero_optimization**: <class 'dict'>

    Default = None

    Configuration for using ZeRO optimization.

    Multi-level dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#zero-optimization-options



- **curriculum_learning**: <class 'dict'>

    Default = None

    



- **curriculum_seqlen**: <class 'int'>

    Default = 0

    Internal var for tracking the current seqlen



- **steps_per_print**: <class 'int'>

    Default = 10

    Print train loss every N steps.



- **wall_clock_breakdown**: <class 'bool'>

    Default = False

    Enable timing of the latency of forward/backward/update training phases.



- **dump_state**: <class 'bool'>

    Default = False

    Print out state information of DeepSpeed object after initialization.



- **flops_profiler**: <class 'dict'>

    Default = None

    Configuration for using FLOPS profiler.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#flops-profiler



- **communication_data_type**: <class 'bool'>

    Default = None

    During gradient averaging, perform communication with selected data type. By default it will be determined by selected regime



- **autotuning**: <class 'dict'>

    Default = None

    Configuration for using autotuning.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#autotuning



- **activation_checkpointing**: <class 'dict'>

    Default = None

    Configuration for using activation checkpointing.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#activation-checkpointing



- **sparse_attention**: <class 'dict'>

    Default = None

    Configuration for using sparse attention.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#sparse-attention



- **data_efficiency**: <class 'dict'>

    Default = None

    Configuration for using data efficiency.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#data-efficiency



- **tensorboard**: <class 'dict'>

    Default = None

    Configuration for using tensorboard.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#monitoring-module-tensorboard-wandb-csv



- **wandb**: <class 'dict'>

    Default = None

    Configuration for using wandb.



- **csv_monitor**: <class 'dict'>

    Default = None

    Configuration for using csv_monitor.



- **elasticity**: <class 'dict'>

    Default = None

    Configuration for using elastic training.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#elastic-training-config-v01-and-v02



- **comms_logger**: <class 'dict'>

    Default = None

    Configuration for using communication logger.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#communication-logging



- **compression_training**: <class 'dict'>

    Default = None

    Configuration for using compression training.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#compression



- **checkpoint**: <class 'dict'>

    Default = None

    Configuration for using checkpointing.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#checkpoint-options



- **data_types**: <class 'dict'>

    Default = None

    Configuration for using data types.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#data-type-options



- **deepspeed_extra_args**: <class 'dict'>

    Default = None

    Dictionary of extra arguments to be included in the yaml config file. This can be used for any argument not included in the above list.



## NeoXArgsDeepspeedRunner

Args for deepspeed runner (deepspeed.launcher.runner).
    Every argument included here will be passed as command line argument to deepspeed.launcher.runner



- **hostfile**: <class 'str'>

    Default = None

    list of hostnames / ssh aliases and the number of GPUs per host

    example file contents:
    worker-1 slots=4
    worker-2 slots=4
    127.0.0 slots=4
    127.0.1 slots=4



- **include**: <class 'str'>

    Default = None

    Specify hardware resources to use during execution. String format is `NODE_SPEC[@NODE_SPEC ...]` where `NODE_SPEC=NAME[:SLOT[,SLOT ...]]`. If `:SLOT` is omitted, include all slots on that host. Example: `"worker-0@worker-1:0,2"` will use all slots. on `worker-0` and slots `[0, 2]` on `worker-1`.



- **exclude**: <class 'str'>

    Default = None

    Specify hardware resources to NOT use during execution. Same format as include



- **num_nodes**: <class 'int'>

    Default = -1

    Total number of worker nodes to run on, this will use the top N hosts from the given hostfile. -1 will use all.



- **num_gpus**: <class 'int'>

    Default = None

    Max number of GPUs to use on each node, will use [0:N) GPU ids on each node. None / not specifying a value will use all.



- **master_port**: <class 'int'>

    Default = 29500

    Port used by PyTorch distributed for communication during training.



- **master_addr**: <class 'str'>

    Default = None

    IP address of node 0, will be inferred via 'hostname -I' if not specified.



- **launcher**: typing.Literal['pdsh', 'openmpi', 'mvapich', 'slurm']

    Default = pdsh

    Launcher backend for multi-node training. Options currently include PDSH, OpenMPI, MVAPICH.



- **force_multi**: <class 'bool'>

    Default = False

    Force multi-node training even if only one node is specified.



- **detect_nvlink_pairs**: <class 'bool'>

    Default = False

    If true, autodetects nvlink pairs and remaps cuda visible devices to place them next to each other. This is an Eleuther addition to deepspeed, and should speed up model parallel training on setups with nvlink pairs when mp=2.



- **autotuning_run**: <class 'str'>

    Default = None

    Either "tune", "run", or `None`.



- **no_ssh_check**: <class 'bool'>

    Default = False

    If true, overrides the default check where DeepSpeed confirms that the headnode is accessible via ssh.



- **comment**: <class 'str'>

    Default = None

    Adds a `--comment` to the DeepSpeed launch command. In DeeperSpeed this is passed on to the SlurmLauncher as well. Sometimes necessary for cluster rules, or so I've heard.



- **account**: <class 'str'>

    Default = None

    Adds a `--account` to the DeepSpeed launch command. In DeeperSpeed this is passed on to the SlurmLauncher as well. Sometimes necessary for cluster rules, or so I've heard.

