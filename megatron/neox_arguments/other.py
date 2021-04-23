from dataclasses import dataclass

@dataclass
class NeoXArgsOther:

    distributed_backend: str = "nccl"
    """
    Which backend to use for distributed training.
    """

    local_rank: int = None
    """
    local rank passed from distributed launcher.
    """

    rank: int = None
    """
    
    """

    lazy_mpu_init: bool = False
    """
    If set to True, initialize_megatron() skips DDP initialization and returns function to complete it instead. Also turns on use-cpu-initialization flag. This is for external DDP manager.
    """

    short_seq_prob: float = 0.1
    """
    Probability of producing a short sequence.
    """

    reset_position_ids: bool = False
    """
    Reset posistion ids after end-of-document token.
    """

    reset_attention_mask: bool = False
    """
    Reset self attention maske after end-of-document token.
    """

    eod_mask_loss: bool = False
    """
    Mask loss for the end of document tokens.
    """

    adlr_autoresume: bool = False
    """
    Enable auto-resume on adlr cluster.
    """

    adlr_autoresume_interval: int = 1000
    """
    Intervals over which check for auto-resume termination signal
    """

    seed: int = 1234
    """
    Random seed used for python, numpy, pytorch, and cuda.
    """

    onnx_safe: bool = False
    """
    Use workarounds for known problems with Torch ONNX exporter
    """

    deepscale: bool = False
    """
    Deprecated enable DeepSpeed (helper flag for user code, no impact on DeepSpeed backend)'
    """

    deepscale_config: str = None
    """Deprecated deepscale json configuration file."""

    deepspeed_mpi: bool = False
    """
    Run via MPI, this will attempt to discover the necessary variables to initialize torch distributed from the MPI environment
    """
    
    user_script: str = None
    """
    user script to be run
    """