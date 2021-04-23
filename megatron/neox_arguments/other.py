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

    lazy_mpu_init: bool = True
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

    prescale_gradients: bool = False
    """
    Scale gradients before doing allreduce.
    """

    gradient_predivide_factor: float = 1.0
    """
    Before gradient averaging predivide gradients by a specified factor, can sometimes help with fp16 stability when scaling to large numbers of GPUs
    """

    sparse_gradients: bool = False
    """
    Enable sparse compression of torch.nn.Embedding gradients.
    """

    amp: dict = None
    """
    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#automatic-mixed-precision-amp-training-options
    """

    flops_profiler: dict = None
    """
    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#flops-profiler
    """
