# Copyright (c) 2024, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


import torch


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(tensor, num_partitions, contiguous_split_chunks=False):
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list

def split_reorder_and_stack_separate_qkv(q, k, v, num_mp_ranks, dim=0, contiguous_qkv_chunks=False):
    """
    Splits separate q, k, v matrices e.g. from huggingface into chunks of size=mp config, then interleaves them so that each worker
    gets its packed qkv matrices appropriately before stacking them.
    Idea (example for GQA):
    q = [q1, q2, q3, q4]
    k = [k1, k2]
    v = [v1, v2]
    1) Split: First split into mp chunks, assuming mp=2 we get   [[q1, q2], [q3, q4]],   [[k1], [k2]],   [[v1], [v2]]
    2) Reorder: Then group relevant qkv for each mp rank: [q1, q2, k1, v1], [q3, q4, k2, v2]
    3) Stack: Consolidate into single qkv: [q1, q2, k1, v1, q3, q4, k2, v2]
    That way when the qkv gets loaded on each rank we avoid [q1, q2, q3, q4] on one rank, [k1, k2, v1, v2] on the other, which would
    be misinterpreted in transformer.py as q3 being a key tensor, q4 being a value tensor, etc.
    
    Relying on the assert happening when mpu.divide gets called when initialising the neox transformer; note this will need to be updated
    if the q, k, v behaviour of transformers.py is changed.

    To perform a simple test on the case num_mp_ranks=2:
    m = 2
    A = torch.ones((8,2))
    B = torch.ones((8,2))*2
    C = torch.ones((8,2))*3
    D = torch.cat([torch.cat((x, y, z), dim=0) for x, y, z in zip(torch.chunk(A, chunks=m, dim=0),
                                                                  torch.chunk(B, chunks=m, dim=0),
                                                                  torch.chunk(C, chunks=m, dim=0))],
                      dim=0)
    """
    def conditional_contiguous(tensor, contiguous_qkv_chunks):
        if contiguous_qkv_chunks:
            return tensor.contiguous()
        else:
            return tensor
    return torch.cat(
                        [
                        conditional_contiguous(torch.cat((x, y, z), dim=dim), contiguous_qkv_chunks) 
                                                    for x, y, z in zip(torch.chunk(q, chunks=num_mp_ranks, dim=dim),
                                                                       torch.chunk(k, chunks=num_mp_ranks, dim=dim),
                                                                       torch.chunk(v, chunks=num_mp_ranks, dim=dim))
                        ],
                        dim=dim
                    )

class VocabUtility:
    """Split the vocabulary into `world_size` chunks amd return the
    first and last index of the vocabulary belonging to the `rank`
    partition: Note that indices in [first, last]"""

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size, rank, world_size
    ):
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size, rank, world_size):
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(
            per_partition_vocab_size, rank, world_size
        )
