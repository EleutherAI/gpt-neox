# Copyright (c) 2021, EleutherAI
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

# mostly moving to using checkpointing from deepspeed (identical code anyway) so currently this file is only imports
# TODO: should be able to get rid of this file entirely

import deepspeed
import deepspeed.runtime.activation_checkpointing.checkpointing as checkpointing

# Default name for the model parallel rng tracker.
_MODEL_PARALLEL_RNG_TRACKER_NAME = (
    deepspeed.checkpointing._MODEL_PARALLEL_RNG_TRACKER_NAME
)

# Whether apply model parallelsim to checkpointed hidden states.
_CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER = None

# RNG tracker object.
_CUDA_RNG_STATE_TRACKER = deepspeed.checkpointing._CUDA_RNG_STATE_TRACKER

# Deepspeed checkpointing functions
# TODO: replace calls to these in our codebase with calls to the deepspeed ones
_set_cuda_rng_state = checkpointing._set_cuda_rng_state
checkpoint = checkpointing.checkpoint
model_parallel_cuda_manual_seed = checkpointing.model_parallel_cuda_manual_seed
get_cuda_rng_tracker = checkpointing.get_cuda_rng_tracker
