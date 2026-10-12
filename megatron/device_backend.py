# Copyright (c) 2026, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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


def _get_backend():
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return torch.cuda

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.xpu

    return None


_backend = _get_backend()


def is_available():
    return _backend is not None


def device():
    if _backend is None:
        return torch.device("cpu")

    if _backend is torch.cuda:
        return torch.device("cuda")

    return torch.device("xpu")


def current_device():
    if _backend is None:
        return 0

    return _backend.current_device()


def device_count():
    if _backend is None:
        return 1

    return _backend.device_count()


def set_device(device_id):
    if _backend is not None:
        _backend.set_device(device_id)


def synchronize(device_id=None):
    if _backend is not None:
        _backend.synchronize(device_id)

def memory(device_id=None):
    if _backend is not None:
        return _backend.memory

def memory_allocated(device_id=None):
    if _backend is None:
        return 0

    return _backend.memory_allocated(device_id)


def max_memory_allocated(device_id=None):
    if _backend is None:
        return 0

    return _backend.max_memory_allocated(device_id)


def memory_reserved(device_id=None):
    if _backend is None:
        return 0

    return _backend.memory_reserved(device_id)

def max_memory_reserved(device_id=None):
    if _backend is None:
        return 0

    return _backend.max_memory_reserved(device_id)

def empty_cache():
    if _backend is not None:
        _backend.empty_cache()


def current_stream(device_id=None):
    if _backend is None:
        return None

    return _backend.current_stream(device_id)


def default_stream(device_id=None):
    if _backend is None:
        return None

    return _backend.default_stream(device_id)


def Event(*args, **kwargs):
    if _backend is None:
        return None

    return _backend.Event(*args, **kwargs)

def get_rng_state():
    if _backend is not None:
        _backend.get_rng_state()

def set_rng_state(state):
    if _backend is not None:
        _backend.set_rng_state(state)

def manual_seed(seed):
    if _backend is not None:
        _backend.manual_seed(seed)


def manual_seed_all(seed):
    if _backend is not None:
        _backend.manual_seed_all(seed)

def nvtx():
    if _backend is torch.cuda:
        return _backend.nvtx()

def cudart():
    if _backend is torch.cuda:
        return _backend.cudart()