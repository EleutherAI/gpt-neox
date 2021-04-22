from dataclasses import dataclass

@dataclass
class NeoXArgsDeepspeedRunnerArguments:    
    #TODO what is this?
    #TODO 'launcher_args' # handle separately: 'user_script', 'user_args'

    deepspeed: bool = True
    """boolean flag to enable DeepSpeed"""

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

    num_gpus: int = -1
    """
    Max number of GPUs to use on each node, will use [0:N) GPU ids on each node. -1 will use all.
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

    