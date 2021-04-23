from dataclasses import dataclass

@dataclass
class NeoXArgsTraining:

    data_path: str = None
    """
    Path to combined dataset to split.
    """

    data_impl: str = 'infer'
    """
    Implementation of indexed datasets.
    """

    mmap_warmup: bool = False
    """
    Warm up mmap files.
    """

    save: str = None
    """
    Output directory to save checkpoints to.
    """

    load: str = None
    """
    Directory containing a model checkpoint.
    """

    save_interval: int = None
    """
    Number of iterations between checkpoint saves.
    """

    no_save_optim: bool = False
    """
    Do not save current optimizer.
    """

    no_save_rng: bool = False
    """
    Do not save current rng state.
    """

    no_load_optim: bool = False
    """
    Do not load optimizer when loading checkpoint.
    """

    no_load_rng: bool = False
    """
    Do not load rng state when loading checkpoint.
    """

    finetune: bool = False
    """
    Load model for finetuning. Do not load optimizer or rng state from checkpoint and set iteration to 0. Assumed when loading a release checkpoint.
    """

    batch_size: int = None
    """
    training batch size
    """

    train_iters: int = None
    """
    Number of iterations to run for training.
    """

    eval_iters: int = 100
    """
    Number of iterations to run for evaluationvalidation/test for.
    """

    log_interval: int = None
    """
    Interval between logging.
    """

    keep_last_n_checkpoints: int = None
    """
    Number of last checkpoints to keep
    """

    eval_interval: int = 1000
    """
    Interval between running evaluation on validation set.
    """

    split: str = "969, 30, 1"
    """
    Comma_separated list of proportions for training, validation, and test split. For example the split 90,5,5 will use 90% of data for training, 5% for validation and 5% for test.
    """

    vocab_file: str = None
    """
    Path to the vocab file.
    """

    merge_file: str = None
    """
    Path to the BPE merge file.
    """

    num_workers: int = 2
    """
    Dataloader number of workers.
    """

    dump_state: bool = False
    """
    Print out state information of DeepSpeed object after initialization.
    """

    exit_interval: int = None
    """
    Exit the program after the iteration is divisible by this value.
    """

    lr_decay_style: str = "linear"
    """
    Learning rate decay function. Choose from 'constant', 'linear', 'cosine', 'exponential'.
    """

    lr_decay_iters: int = None
    """
    Number of iterations to decay learning rate over, If None defaults to --train-iters
    """

    min_lr: float = 0.0
    """
    Minumum value for learning rate. The scheduler clips values below this threshold.
    """

    warmup: float = 0.01
    """
    Percentage of total iterations to warmup on (.01 = 1 percent of all training iters).
    """

    override_lr_scheduler: bool = False
    """
    Reset the values of the scheduler (learning rate,warmup iterations, minimum learning rate, maximum number of iterations, and decay style from input arguments and ignore values from checkpoints. Note that all the above values will be reset.
    """

    use_checkpoint_lr_scheduler: bool = False
    """
    Use checkpoint to set the values of the scheduler (learning rate, warmup iterations, minimum learning rate, maximum number of iterations, and decay style from checkpoint and ignore input arguments.
    """

    attention_dropout: float = 0.1
    """
    Post attention dropout probability.
    """

    hidden_dropout: float = 0.1
    """
    Dropout probability for hidden state transformer.
    """

    weight_decay: float = 0.01
    """
    Weight decay coefficient for L2 regularization.
    """

    checkpoint_activations: bool = False
    """
    Checkpoint activation to allow for training with larger models, sequences, and batch sizes.
    """

    checkpoint_num_layers: int = 1
    """
    Chunk size (number of layers) for checkpointing.
    """

    distribute_checkpointed_activations: bool = False
    """
    If set, distribute checkpointed activations across model parallel group.
    """

    deepspeed_activation_checkpointing: bool = False
    """
    Uses activation checkpointing from deepspeed
    """
    
    contiguous_checkpointing: bool = False
    """
    Contiguous memory checkpointing for activations.
    """
    
    checkpoint_in_cpu: bool = False
    """
    Move the activation checkpoints to CPU.
    """
    
    synchronize_each_layer: bool = False
    """
    does a synchronize at the beginning and end of each checkpointed layer.
    """
    
    profile_backward: bool = False
    """
    Enables backward pass profiling for checkpointed layers.
    """
    
    partition_activations: bool = False
    """
    Partition Activations across GPUs before checkpointing.
    """

    gas: int = None
    """gradient_accumulation_steps""" #TODO this is a duplicate, remove?


    clip_grad: float = None
    """
    Gradient clipping based on global L2 norm.
    """

    no_adamw: bool = False
    """
    Use default Adam instead of AdamW
    """

    sm3: bool = False
    """
    Enable sm3 optimizer
    """

    hysteresis: int = 2
    """
    hysteresis for dynamic loss scaling
    """

    dynamic_loss_scale: bool = None
    """
    flag indicating whether dynamic loss scale is used
    """

    loss_scale: float = None
    """
    Static loss scaling, positive power of 2
    values can improve fp16 convergence. If None, dynamic loss scaling is used.
    """

    loss_scale_window: float = 1000.0
    """
    Window over which to raise/lower dynamic scale.
    """

    min_scale: float = 1.0
    """
    Minimum loss scale for dynamic loss scale.
    """