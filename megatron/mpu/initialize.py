# Copyright (c) 2021, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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


"""Model and data parallel groups."""

import torch

from .utils import ensure_divisibility

# Model parallel group that the current rank belongs to.
_TENSOR_PARALLEL_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
# Pipeline parallel group that the current rank belongs to.
_PIPE_PARALLEL_GROUP = None
# Sequence parallel group that the current rank belongs to.
_SEQUENCE_PARALLEL_GROUP = None

# A group used to sync during the IO process. Usually this is data_parallel_group(),
# but with pipeline parallelism it must also involve the last stage (which is not in the
# DP group of rank 0)
_IO_PARALLEL_GROUP = None

# These values enable us to change the mpu sizes on the fly.
_MPU_OR_SPU_WORLD_SIZE = None
_MPU_OR_SPU_RANK = None

# These values enable us to change the mpu sizes on the fly.
_MPU_OR_SPU_WORLD_SIZE = None
_MPU_OR_SPU_RANK = None

# Used to query 3D topology
_MPU_OR_SPU_TOPOLOGY = None

# Get fp32_allreduce flag
_FP32_ALLREDUCE = None

# Are we using deepspeed sequence parallelism
_IS_SEQUENCE_PARALLEL = None

def is_uninitialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is None


def initialize_model_parallel(tensor_model_parallel_size, sequence_model_parallel_size, pipeline_model_parallel_size, neox_args, fp32_allreduce=False):
    """
    Initialize model data parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used to parallelize model.
        pipeline_model_parallel_size: number of GPUs used to parallelize model.
        sequence_model_parallel_size: number of GPUs used to parallelize model.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model. The present function will
    create 4 model parallel groups and 2 data parallel groups as:
        4 model parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 data parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """

    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    global _IS_SEQUENCE_PARALLEL
    _IS_SEQUENCE_PARALLEL = sequence_model_parallel_size > 1

    if _IS_SEQUENCE_PARALLEL:
        assert tensor_model_parallel_size == 1 and pipeline_model_parallel_size == 1, \
        'DeepSpeed\'s sequence parallel does not work with tensor parallel or pipeline parallel'

    if torch.distributed.get_rank() == 0:
        if _IS_SEQUENCE_PARALLEL:
            print("> initializing sequence model parallel with size {}".format(sequence_model_parallel_size))
        else:
            print("> initializing tensor model parallel with size {}, and pipeline model parallel with size {}".format(tensor_model_parallel_size, pipeline_model_parallel_size))

    if _IS_SEQUENCE_PARALLEL:
        # Ensure none of the parallel sizes are too large
        if world_size < sequence_model_parallel_size:
            raise ValueError("world size cannot be smaller than sequence model parallel size")
        # Ensure each axis is divisible by world size
        ensure_divisibility(world_size, sequence_model_parallel_size)
        data_parallel_size = world_size // sequence_model_parallel_size
    else:
        # Ensure none of the parallel sizes are too large
        if world_size < tensor_model_parallel_size * pipeline_model_parallel_size:
            raise ValueError("world size cannot be smaller than tensor_model_parallel_size * pipeline_model_parallel_size")
        # Ensure each axis is divisible by world size
        ensure_divisibility(world_size, tensor_model_parallel_size)
        ensure_divisibility(world_size, pipeline_model_parallel_size)
        data_parallel_size = world_size // (tensor_model_parallel_size * pipeline_model_parallel_size)


    # Set up the topology
    from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology, ProcessTopology

    if _IS_SEQUENCE_PARALLEL:
        topology = ProcessTopology(axes=['data', 'sequence'], dims=[data_parallel_size, sequence_model_parallel_size])
    else:
        # this does pipe on the most outside, then data, then model.
        # PipeModelDataParallelTopology is just a wrapper over ProcessTopology that predefines this order.
        topology = PipeModelDataParallelTopology(num_pp=pipeline_model_parallel_size, num_mp=tensor_model_parallel_size, num_dp=data_parallel_size)

        # Offset base seeds for the interior pipeline stages.
        # TODO: adjust last stage too once IO is improved.
        stage_id = topology.get_coord(rank=torch.distributed.get_rank()).pipe
        if 0 < stage_id < topology.get_dim("pipe") - 1:
            offset = neox_args.seed + 1138
            neox_args.seed = offset + (stage_id * mp)

    #data_parallel_size: int = world_size // (tensor_model_parallel_size * pipeline_model_parallel_size * sequence_parallel_size)
    #sequence_data_parallel_size: int = sequence_parallel_size * data_parallel_size

    #assert (sequence_parallel_size > 1 and tensor_model_parallel_size == 1) or (sequence_parallel_size == 1), "sequence parallelism not yet supported with pipeline or tensor parallelism"
    #num_sequence_parallel_groups: int = world_size // sequence_parallel_size
    #num_sequence_data_parallel_groups: int = world_size // sequence_parallel_size // data_parallel_size

    rank = torch.distributed.get_rank()

    global _MPU_OR_SPU_TOPOLOGY
    if topology:
        _MPU_OR_SPU_TOPOLOGY = topology

    # Build the data parallel groups.
    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, "data parallel group is already initialized"
    if topology:
        for dp_group in topology.get_axis_comm_lists("data"):
            group = torch.distributed.new_group(ranks=dp_group)
            if rank == 0:
                print(f"DP Group:", dp_group)
            if rank in dp_group:
                _DATA_PARALLEL_GROUP = group
    else:
        if _IS_SEQUENCE_PARALLEL:
            for i in range(sequence_model_parallel_size):
                ranks = range(i, world_size, sequence_model_parallel_size)
                group = torch.distributed.new_group(ranks)
                if i == (rank % sequence_model_parallel_size):
                    _DATA_PARALLEL_GROUP = group
        else:
            for i in range(tensor_model_parallel_size):
                ranks = range(i, world_size, tensor_model_parallel_size)
                group = torch.distributed.new_group(ranks)
                if i == (rank % tensor_model_parallel_size):
                    _DATA_PARALLEL_GROUP = group

    # Build pipeline parallel group
    if topology is not None and not _IS_SEQUENCE_PARALLEL:
        global _PIPE_PARALLEL_GROUP
        for pp_group in topology.get_axis_comm_lists("pipe"):
            group = torch.distributed.new_group(ranks=pp_group)
            if rank == 0:
                print(f"PP Group:", pp_group)
            if rank in pp_group:
                _PIPE_PARALLEL_GROUP = group

    # Build IO group for PP
    global _IO_PARALLEL_GROUP
    if topology and not _IS_SEQUENCE_PARALLEL and topology.get_dim("pipe") > 1:
        io_stages = [0, topology.get_dim("pipe") - 1]
        io_group = []
        for stage in io_stages:
            io_group.extend(topology.filter_match(pipe=stage, model=0))
        if rank == 0:
            print(f"MPU IO:", io_group)
        group = torch.distributed.new_group(ranks=io_group)
        if rank in io_group:
            _IO_PARALLEL_GROUP = group
    else:
        _IO_PARALLEL_GROUP = get_data_parallel_group()

    # Build the model parallel groups.
    global _SEQUENCE_PARALLEL_GROUP
    assert _SEQUENCE_PARALLEL_GROUP is None, "sequence model parallel group is already initialized"
    global _TENSOR_PARALLEL_GROUP
    assert _TENSOR_PARALLEL_GROUP is None, "tensor model parallel group is already initialized"
    if topology:
        # Short circuit case without tensor/sequence parallelism.
        # TODO: it would be nice  to avoid this branching case?
        if _IS_SEQUENCE_PARALLEL:
            if sequence_model_parallel_size == 1:
                for group_rank in range(world_size):
                    group = torch.distributed.new_group(ranks=[group_rank])
                    if rank == 0:
                        print(f"SP Group:", [group_rank])
                    if rank == group_rank:
                        _SEQUENCE_PARALLEL_GROUP = group
                return
        else:
            if tensor_model_parallel_size == 1:
                for group_rank in range(world_size):
                    group = torch.distributed.new_group(ranks=[group_rank])
                    if rank == 0:
                        print(f"TP Group:", [group_rank])
                    if rank == group_rank:
                        _TENSOR_PARALLEL_GROUP = group
            return

        if _IS_SEQUENCE_PARALLEL:
            for sp_group in topology.get_axis_comm_lists("sequence"):
                group = torch.distributed.new_group(ranks=sp_group)
                if rank == 0:
                    print(f"SP Group:", sp_group)
                if rank in sp_group:
                    _SEQUENCE_PARALLEL_GROUP = group
        else:
            for tp_group in topology.get_axis_comm_lists("model"):
                group = torch.distributed.new_group(ranks=tp_group)
                if rank == 0:
                    print(f"TP Group:", tp_group)
                if rank in tp_group:
                    _TENSOR_PARALLEL_GROUP = group
    else:
        if _IS_SEQUENCE_PARALLEL:
            for i in range(world_size // sequence_model_parallel_size):
                ranks = range(i * sequence_model_parallel_size, (i + 1) * sequence_model_parallel_size)
                group = torch.distributed.new_group(ranks)
                if i == (rank // sequence_model_parallel_size):
                    _SEQUENCE_PARALLEL_GROUP = group
        else:
            for i in range(world_size // tensor_model_parallel_size):
                ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
                group = torch.distributed.new_group(ranks)
                if i == (rank // tensor_model_parallel_size):
                    _TENSOR_PARALLEL_GROUP = group

    global _FP32_ALLREDUCE
    assert _FP32_ALLREDUCE is None, "fp32_allreduce is already initialized"
    _FP32_ALLREDUCE = fp32_allreduce

# Check if initialized
def tensor_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if _TENSOR_PARALLEL_GROUP is None or _DATA_PARALLEL_GROUP is None:
        return False
    return True

def sequence_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if _SEQUENCE_PARALLEL_GROUP is None or _DATA_PARALLEL_GROUP is None:
        return False
    return True


# Get the parallel group
def get_sequence_model_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    return _SEQUENCE_PARALLEL_GROUP

def get_tensor_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _TENSOR_PARALLEL_GROUP is not None, "model parallel group is not initialized"
    return _TENSOR_PARALLEL_GROUP

def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, "data parallel group is not initialized"
    return _DATA_PARALLEL_GROUP

def get_pipe_model_parallel_group():
    """Get the pipe parallel group the caller rank belongs to."""
    assert _PIPE_PARALLEL_GROUP is not None, "data parallel group is not initialized"
    return _PIPE_PARALLEL_GROUP

def get_io_parallel_group():
    """Get the IO parallel group the caller rank belongs to."""
    assert _IO_PARALLEL_GROUP is not None, "IO parallel group is not initialized"
    return _IO_PARALLEL_GROUP


# Set the parallel world size
def set_tensor_model_parallel_world_size(world_size):
    """Set the model parallel size"""
    global _MPU_OR_SPU_WORLD_SIZE
    _MPU_OR_SPU_WORLD_SIZE = world_size

def set_sequence_model_parallel_world_size(world_size):
    """Set the model parallel size"""
    global _MPU_OR_SPU_WORLD_SIZE
    _MPU_OR_SPU_WORLD_SIZE = world_size


# Get the parallel world size
def get_tensor_model_parallel_world_size():
    """Return world size for the model parallel group."""
    global _MPU_OR_SPU_WORLD_SIZE
    if _MPU_OR_SPU_WORLD_SIZE is not None:
        return _MPU_OR_SPU_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())

def get_sequence_model_parallel_world_size():
    """Return world size for the model parallel group."""
    global _MPU_OR_SPU_WORLD_SIZE
    if _MPU_OR_SPU_WORLD_SIZE is not None:
        return _MPU_OR_SPU_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_sequence_model_parallel_group())


# Set the parallel rank
def set_tensor_model_parallel_rank(rank):
    """Set tensor parallel rank."""
    global _MPU_OR_SPU_RANK
    _MPU_OR_SPU_RANK = rank

def set_sequence_model_parallel_rank(rank):
    """Set sequence parallel rank."""
    global _MPU_OR_SPU_RANK
    _MPU_OR_SPU_RANK = rank


# Get the parallel rank
def get_tensor_model_parallel_rank():
    """Return my rank for the model parallel group."""
    global _MPU_OR_SPU_RANK
    if _MPU_OR_SPU_RANK is not None:
        return _MPU_OR_SPU_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())

def get_sequence_model_parallel_rank():
    """Return my rank for the model parallel group."""
    global _MPU_OR_SPU_RANK
    if _MPU_OR_SPU_RANK is not None:
        return _MPU_OR_SPU_RANK
    return torch.distributed.get_rank(group=get_sequence_model_parallel_group())

def get_pipe_model_parallel_rank():
    """Return my rank for the pipe parallel group."""
    return torch.distributed.get_rank(group=get_pipe_model_parallel_group())


# Get the src rank
def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to a local rank zero
    in the model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size

def get_sequence_model_parallel_src_rank():
    """Calculate the global rank corresponding to a local rank zero
    in the model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_sequence_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size

def get_data_parallel_src_rank():
    """Calculate the global rank corresponding to a local rank zero
    in the data parallel group."""
    global_rank = torch.distributed.get_rank()
    topo = get_topology()
    if _IS_SEQUENCE_PARALLEL:
        # we are just using tensor parallel
        return global_rank % get_tensor_model_parallel_world_size()
    else:
        if topo is None:
            # we are just using tensor parallel
            return global_rank % get_tensor_model_parallel_world_size()
        else:
            # We are using pipeline parallel
            d = topo.get_axis_comm_lists("data")
            for l in d:
                if global_rank in l:
                    return l[0]

# Get the world size
def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_data_parallel_group())

def get_pipe_parallel_world_size():
    """Return world size for the pipe parallel group."""
    return torch.distributed.get_world_size(group=get_pipe_model_parallel_group())

def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(group=get_data_parallel_group())

def is_sequence_parallel():
    """Return whether sequence parallelism is used"""
    return _IS_SEQUENCE_PARALLEL

# Get topology
def get_topology():
    return _MPU_OR_SPU_TOPOLOGY


def destroy_model_parallel():
    """Set the groups to none."""
    global _TENSOR_PARALLEL_GROUP
    _TENSOR_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _PIPE_PARALLEL_GROUP
    _PIPE_PARALLEL_GROUP = None
    global _SEQUENCE_PARALLEL_GROUP
    _SEQUENCE_PARALLEL_GROUP = None
    global _IO_PARALLEL_GROUP
    _IO_PARALLEL_GROUP = None
    global _MPU_OR_SPU_WORLD_SIZE
    global _MPU_OR_SPU_RANK
    _MPU_OR_SPU_WORLD_SIZE = None
    _MPU_OR_SPU_RANK = None
    global _MPU_OR_SPU_TOPOLOGY
    _MPU_OR_SPU_TOPOLOGY = None
    global _FP32_ALLREDUCE
    _FP32_ALLREDUCE = None


def get_fp32_allreduce():
    """Get the fp32 allreduce flag"""
    assert _FP32_ALLREDUCE is not None, "fp32_allreduce is not Initialized"
    return _FP32_ALLREDUCE
