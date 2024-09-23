# Copyright (c) 2024, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

try:
    from .template import NeoXArgsTemplate
except ImportError:
    from template import NeoXArgsTemplate

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


@dataclass
class NeoXArgsDeepspeedConfig(NeoXArgsTemplate):
    """
    Args for deepspeed config
    Every argument included here will be included in deepspeed config json
    As of Mar 8 2023, up to date compared to https://www.deepspeed.ai/docs/config-json/
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

    # ---FP16 Training Options---

    fp16: dict = None
    """
    Configuration for using mixed precision/FP16 training that leverages NVIDIA’s Apex package.

    Dictionary options as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#fp16-training-options
    """

    bf16: dict = None
    """
    Configuration for using bfloat16 floating-point format as an alternative to FP16. BFLOAT16 requires hardware support (e.g., NVIDIA A100).

    Dictionary options as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#bfloat16-training-options
    """

    # ---Automatic Mixed Precision (AMP) Training Options---

    amp: dict = None
    """
    Configuration for using automatic mixed precision (AMP) training that leverages NVIDIA’s Apex AMP package.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#automatic-mixed-precision-amp-training-options
    """

    gradient_clipping: float = 1.0
    """
    Enable gradient clipping with provided value
    """

    # ---ZeRO Optimization Options---

    zero_optimization: dict = None
    """
    Configuration for using ZeRO optimization.

    Multi-level dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#zero-optimization-options
    """

    # ---Logging Options---

    curriculum_learning: dict = None
    """"""

    curriculum_seqlen: int = 0
    """
    Internal var for tracking the current seqlen
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

    # ---FLOPS Profiler Options---

    flops_profiler: dict = None
    """
    Configuration for using FLOPS profiler.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#flops-profiler
    """

    # ---Communication Options---

    communication_data_type: bool = None
    """
    During gradient averaging, perform communication with selected data type. By default it will be determined by selected regime
    """

    # ---Autotuning Options---
    autotuning: dict = None
    """
    Configuration for using autotuning.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#autotuning
    """

    # ---Activation Checkpointing Options---

    activation_checkpointing: dict = None
    """
    Configuration for using activation checkpointing.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#activation-checkpointing
    """

    # ---Sparse Attention Options---

    sparse_attention: dict = None
    """
    Configuration for using sparse attention.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#sparse-attention

    """

    # ---Data Efficiency Options---

    data_efficiency: dict = None
    """
    Configuration for using data efficiency.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#data-efficiency
    """

    # ---Monitoring Module Options---

    tensorboard: dict = None
    """
    Configuration for using tensorboard.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#monitoring-module-tensorboard-wandb-csv
    """

    wandb: dict = None
    """
    Configuration for using wandb.
    """

    csv_monitor: dict = None
    """
    Configuration for using csv_monitor.
    """

    # ---Elastic Training Options---

    elasticity: dict = None
    """
    Configuration for using elastic training.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#elastic-training-config-v01-and-v02
    """

    # ---Communication Logging Options---

    comms_logger: dict = None
    """
    Configuration for using communication logger.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#communication-logging
    """

    # ---Compression Options---

    compression_training: dict = None
    """
    Configuration for using compression training.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#compression
    """

    # ---Checkpointing Options---

    checkpoint: dict = None
    """
    Configuration for using checkpointing.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#checkpoint-options
    """

    # ---Data Type Options---

    data_types: dict = None
    """
    Configuration for using data types.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#data-type-options
    """

    # ---EXTRA ARGUMENTS---

    deepspeed_extra_args: dict = None
    """
    Dictionary of extra arguments to be included in the yaml config file. This can be used for any argument not included in the above list.
    """

    autotuning: dict = None
    """Dictionary as described in DeepSpeed autotuning documentation: https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/autotuning"""


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

    launcher: Literal["pdsh", "openmpi", "mvapich", "slurm"] = "pdsh"
    """
    Launcher backend for multi-node training. Options currently include PDSH, OpenMPI, MVAPICH.
    """

    force_multi: bool = False
    """
    Force multi-node training even if only one node is specified.
    """

    detect_nvlink_pairs: bool = False
    """
    If true, autodetects nvlink pairs and remaps cuda visible devices to place them next to each other. This is an Eleuther addition to deepspeed, and should speed up model parallel training on setups with nvlink pairs when mp=2.
    """

    autotuning_run: str = None
    """
    Either "tune", "run", or `None`.
    """

    no_ssh_check: bool = False
    """
    If true, overrides the default check where DeepSpeed confirms that the headnode is accessible via ssh.
    """

    force_multi: bool = False
    """
    If true, Force multi-node launcher mode, helps in cases where user wants to launch on single remote node.
    """

    comment: str = None
    """
    Adds a `--comment` to the DeepSpeed launch command. In DeeperSpeed this is passed on to the SlurmLauncher as well. Sometimes necessary for cluster rules, or so I've heard.
    """

    account: str = None
    """
    Adds a `--account` to the DeepSpeed launch command. In DeeperSpeed this is passed on to the SlurmLauncher as well. Sometimes necessary for cluster rules, or so I've heard.
    """
