from dataclasses import dataclass

try:
    from .template import NeoXArgsTemplate
except ImportError:
    from template import NeoXArgsTemplate


@dataclass
class NeoXArgsDeepspeedConfig(NeoXArgsTemplate):
    """
    Args for deepspeed config
    Every argument included here will be included in deepspeed config json
    #TODO this list is not complete as compared to https://www.deepspeed.ai/docs/config-json/
    """

    deepspeed: bool = True
    """boolean flag to enable DeepSpeed (Always True)"""

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

    zero_allow_untested_optimizer: bool = False
    """
    Whether Deepspeed Zero Optimizer will allow an optimizer that hasn't been tested by the deepspeed team
    """


@dataclass
class NeoXArgsDeepspeedRunner(NeoXArgsTemplate):
    """
    Args for deepspeed runner (deepspeed.launcher.runner).
    Every argument included here will be passed as command line argument to deepspeed.launcher.runner
    """

    hostfile: str = None
    """
    list of hostnames / ssh aliases and the number of GPUs per host

    example file contents:
    worker-1 slots=4
    worker-2 slots=4
    127.0.0 slots=4
    127.0.1 slots=4
    """

    include: str = None
    """
    Specify hardware resources to use during execution. String format is `NODE_SPEC[@NODE_SPEC ...]` where `NODE_SPEC=NAME[:SLOT[,SLOT ...]]`. If `:SLOT` is omitted, include all slots on that host. Example: `"worker-0@worker-1:0,2"` will use all slots. on `worker-0` and slots `[0, 2]` on `worker-1`.
    """

    exclude: str = None
    """
    Specify hardware resources to NOT use during execution. Same format as include
    """

    num_nodes: int = -1
    """
    Total number of worker nodes to run on, this will use the top N hosts from the given hostfile. -1 will use all.
    """

    num_gpus: int = None
    """
    Max number of GPUs to use on each node, will use [0:N) GPU ids on each node. None / not specifying a value will use all.
    """

    master_port: int = 29500
    """
    Port used by PyTorch distributed for communication during training.
    """

    master_addr: str = None
    """
    IP address of node 0, will be inferred via 'hostname -I' if not specified.
    """

    launcher: str = "pdsh"
    """
    Launcher backend for multi-node training. Options currently include PDSH, OpenMPI, MVAPICH.
    """

    detect_nvlink_pairs: bool = False
    """
    If true, autodetects nvlink pairs and remaps cuda visible devices to place them next to each other. This is an Eleuther addition to deepspeed, and should speed up model parallel training on setups with nvlink pairs when mp=2.
    """

    slurm_comment: str = None
    """
    If using SLURM launcher adds a `--comment` to the srun command that launches the job. Sometimes necessary for cluster rules, or so I've heard.
    """
