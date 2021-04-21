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

    seed: int = 1234
    """
    Random seed used for python, numpy, pytorch, and cuda.
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

    eval_iters: int = 100
    """
    Number of iterations to run for evaluationvalidation/test for.
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

    log_dir: str = None
    """
    Directory to save logs to.
    """

    tensorboard_dir: str = None
    """
    Write TensorBoard logs to this directory.
    """

    num_workers: int = 2
    """
    Dataloader number of workers.
    """

    steps_per_print: int = 10
    """
    Print train loss every N steps.
    """

    wall_clock_breakdown: bool = False
    """
    Enable timing of the latency of forward/backward/update training phases.
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

    gradient_clipping: float = 0
    """
    Enable gradient clipping with provided value
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


