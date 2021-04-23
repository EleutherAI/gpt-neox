from dataclasses import dataclass

@dataclass
class NeoXArgsDeepspeedConfig:
    """
    args for deepspeed config
    Every argument included here will be included in deepspeed config json
    """ 

    deepspeed: bool = True
    """boolean flag to enable DeepSpeed"""
    
    train_batch_size: int = None
    """"""

    train_micro_batch_size_per_gpu: int = None
    """"""

    gradient_accumulation_steps: int = None
    """"""

    optimizer: dict = None
    """"""

    scheduler: dict = None
    """"""

    fp16: dict = None
    """"""

    gradient_clipping: float = 0.0
    """
    Enable gradient clipping with provided value
    """

    zero_optimization: dict = None
    """"""

    steps_per_print: int = 10
    """
    Print train loss every N steps.
    """

    wall_clock_breakdown: bool = False
    """
    Enable timing of the latency of forward/backward/update training phases.
    """

    steps_per_print: int = None
    """"""

    fp32_allreduce: bool = False
    """
    During gradient averaging perform allreduce with 32 bit values
    """

    #TODO ['prescale_gradients', 'gradient_predivide_factor', 'sparse_gradients',
    #              'amp', 
    #              'dump_state', 'flops_profiler', 'activation_checkpointing', 'sparse_attention',
    #              'zero_allow_untested_optimizer', ]