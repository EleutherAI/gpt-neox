from dataclasses import dataclass

@dataclass
class NeoXArgsDeepspeedConfig:
    """
    args for deepspeed config
    Every argument included here will be included in deepspeed config json
    #TODO this list is not complete as compared to https://www.deepspeed.ai/docs/config-json/
    """ 

    deepspeed: bool = True
    """boolean flag to enable DeepSpeed"""
    
    train_batch_size: int = None
    """
    The effective training batch size. This is the amount of data samples that leads to one step of model update. train_batch_size is aggregated by the batch size that a single GPU processes in one forward/backward pass (a.k.a., train_step_batch_size), the gradient accumulation steps (a.k.a., gradient_accumulation_steps), and the number of GPUs.
    """

    train_micro_batch_size_per_gpu: int = None
    """
    Batch size to be processed by one GPU in one step (without gradient accumulation). When specified, gradient_accumulation_steps is automatically calculated using train_batch_size and number of GPUs. Should not be concurrently specified with gradient_accumulation_steps in the configuration JSON.
    """

    gradient_accumulation_steps: int = 1
    """
    Number of training steps to accumulate gradients before averaging and applying them. This feature is sometimes useful to improve scalability since it results in less frequent communication of gradients between steps. Another impact of this feature is the ability to train with larger batch sizes per GPU. When specified, train_step_batch_size is automatically calculated using train_batch_size and number of GPUs. Should not be concurrently specified with train_step_batch_size in the configuration JSON.
    """

    optimizer: dict = None
    """
    dict containing the keys type and params

    type: The optimizer name. DeepSpeed natively supports Adam, AdamW, OneBitAdam, Lamb, and OneBitLamb optimizers (See here for details) and will import other optimizers from torch.

    params: Dictionary of parameters to instantiate optimizer. The parameter names must match the optimizer constructor signature (e.g., for Adam).
    """

    scheduler: dict = None
    """
    dict containing the keys type and params

    type: The scheduler name. See here (https://deepspeed.readthedocs.io/en/latest/schedulers.html) for list of support schedulers.
    
    params: Dictionary of parameters to instantiate scheduler. The parameter names should match scheduler constructor signature.
    """

    fp32_allreduce: bool = False
    """
    During gradient averaging perform allreduce with 32 bit values
    """

    prescale_gradients: bool = False
    """
    Scale gradients before doing allreduce
    """

    gradient_predivide_factor: float = 1.0
    """
    Before gradient averaging predivide gradients by a specified factor, can sometimes help with fp16 stability when scaling to large numbers of GPUs
    """

    sparse_gradients: bool = False
    """
    Enable sparse compression of torch.nn.Embedding gradients.
    """

    fp16: dict = None
    """
    Configuration for using mixed precision/FP16 training that leverages NVIDIA’s Apex package. 
    """

    amp: dict = None
    """
    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#automatic-mixed-precision-amp-training-options
    """

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

    dump_state: bool = False
    """
    Print out state information of DeepSpeed object after initialization.
    """

    flops_profiler: dict = None
    """
    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#flops-profiler
    """

    activation_checkpointing: dict = None
    """
    """

    sparse_attention: dict = None
    """
    """

    zero_allow_untested_optimizer: bool = False
    """
    """