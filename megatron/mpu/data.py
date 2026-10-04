# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from megatron import device_backend

from .initialize import get_model_parallel_group
from .initialize import get_model_parallel_rank
from .initialize import get_model_parallel_src_rank
from .initialize import get_context_parallel_world_size, get_context_parallel_rank


_MAX_DATA_DIM = 4


def _check_data_types(keys, data, target_dtype):
    """Check that all the keys have the same target data type."""
    for key in keys:
        assert (
            data[key].dtype == target_dtype
        ), "{} has data type {} which " "is different than {}".format(
            key, data[key].dtype, target_dtype
        )


def _build_key_size_numel_dictionaries(keys, data):
    """Build the size on rank 0 and broadcast."""
    max_dim = _MAX_DATA_DIM
    sizes = [0 for _ in range(max_dim) for _ in keys]

    # Pack the sizes on rank zero.
    if get_model_parallel_rank() == 0:
        offset = 0
        for key in keys:
            assert data[key].dim() < max_dim, "you should increase MAX_DATA_DIM"
            size = data[key].size()
            for i, s in enumerate(size):
                sizes[i + offset] = s
            offset += max_dim

    # Move to GPU and broadcast.
    sizes_cuda = torch.LongTensor(sizes).to(device_backend.device())
    torch.distributed.broadcast(
        sizes_cuda, get_model_parallel_src_rank(), group=get_model_parallel_group()
    )

    # Move back to cpu and unpack.
    sizes_cpu = sizes_cuda.cpu()
    key_size = {}
    key_numel = {}
    total_numel = 0
    offset = 0
    for key in keys:
        i = 0
        size = []
        numel = 1
        while sizes_cpu[offset + i] > 0:
            this_size = sizes_cpu[offset + i]
            size.append(this_size)
            numel *= this_size
            i += 1
        key_size[key] = size
        key_numel[key] = numel
        total_numel += numel
        offset += max_dim

    return key_size, key_numel, total_numel


def broadcast_data(keys, data, datatype):
    """Broadcast data from rank zero of each model parallel group to the
    members of the same model parallel group.

    Arguments:
        keys: list of keys in the data dictionary to be broadcasted
        data: data dictionary of string keys and cpu tensor values.
        datatype: torch data type of all tensors in data associated
                  with keys.
    """
    # Build (key, size) and (key, number of elements) dictionaries along
    # with the total number of elements on all ranks.
    key_size, key_numel, total_numel = _build_key_size_numel_dictionaries(keys, data)

    # Pack on rank zero.
    if get_model_parallel_rank() == 0:
        # Check that all keys have the same data type.
        _check_data_types(keys, data, datatype)
        # Flatten the data associated with the keys
        flatten_data = torch.cat(
            [data[key].contiguous().view(-1) for key in keys], dim=0
        ).to(device_backend.device())
    else:
        flatten_data = torch.empty(
            total_numel, device=device_backend.current_device(), dtype=datatype
        )

    # Broadcast
    torch.distributed.broadcast(
        flatten_data, get_model_parallel_src_rank(), group=get_model_parallel_group()
    )

    # Unpack
    output = {}
    offset = 0
    for key in keys:
        size = key_size[key]
        numel = key_numel[key]
        output[key] = flatten_data.narrow(0, offset, numel).view(size)
        offset += numel

    return output

def scatter_data(tensor, zigzag):
    worldsize = get_context_parallel_world_size()
    if worldsize <= 1:
        return tensor
    if zigzag:
        return torch.chunk(tensor, worldsize, dim=-1)[get_context_parallel_rank()]
    # otherwise prepare for zigzagging
    seq_chunks = torch.chunk(tensor, 2 * worldsize, dim=-1)
    data = [
        torch.cat((seq_chunks[i], seq_chunks[-(i + 1)]), dim=-1)
        for i in range(worldsize)
    ]
    return data[get_context_parallel_rank()].contiguous()

    '''
    if get_context_parallel_world_size() <= 1:
        return tokens, position_ids, attention_mask, labels, loss_mask
    cp_size = get_context_parallel_world_size()

    if get_context_parallel_rank() == 0:
        tokens = tokens.flatten() 
        position_ids = position_ids.flatten() 
        attention_mask = attention_mask.flatten()
        labels = labels.flatten()
        loss_mask = loss_mask.flatten()
    
    scattered_tokens = torch.empty(
            tokens.numel()//cp_size, device=device_backend.current_device(), dtype=tokens.datatype()
        )
    scattered_position_ids = torch.empty(
            position_ids.numel()//cp_size, device=device_backend.current_device(), dtype=position_ids.datatype()
        )
    scattered_attention_mask = torch.empty(
            attention_mask.numel()//cp_size, device=device_backend.current_device(), dtype=attention_mask.datatype()
        )
    scattered_labels = torch.empty(
            labels.numel()//cp_size, device=device_backend.current_device(), dtype=labels.datatype()
        )
    scattered_loss_mask = torch.empty(
            loss_mask.numel()//cp_size, device=device_backend.current_device(), dtype=loss_mask.datatype()
        )

    tokens = torch.chunk(tokens, cp_size, dim=0)
    position_ids = torch.chunk(position_ids, cp_size, dim=0)
    attention_mask = torch.chunk(attention_mask, cp_size, dim=0)
    labels = torch.chunk(labels, cp_size, dim=0)
    loss_mask = torch.chunk(loss_mask, cp_size, dim=0)

    torch.distributed.scatter(
            scattered_tokens, scatter_list=tokens, src=get_context_parallel_src_rank(), group=get_context_parallel_group()
        )
    torch.distributed.scatter(
            scattered_position_ids, scatter_list=position_ids, src=get_context_parallel_src_rank(), group=get_context_parallel_group()
        )
    torch.distributed.scatter(
            scattered_attention_mask, scatter_list=attention_mask, src=get_context_parallel_src_rank(), group=get_context_parallel_group()
        )
    torch.distributed.scatter(
            scattered_labels, scatter_list=labels, src=get_context_parallel_src_rank(), group=get_context_parallel_group()
        )
    torch.distributed.scatter(
            scattered_loss_mask, scatter_list=loss_mask, src=get_context_parallel_src_rank(), group=get_context_parallel_group()
        )
    return scattered_tokens, scattered_position_ids, scattered_attention_mask, scattered_labels, scattered_loss_mask
    

'''