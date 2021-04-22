from dataclasses import dataclass

@dataclass
class NeoXArgsDeepspeedConfig:

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

    gradient_clipping: float = 0
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

    steps_per_print: int = None
    """"""

    steps_per_print: int = None
    """"""
