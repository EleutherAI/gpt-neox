# mostly moving to using checkpointing from deepspeed (identical code anyway) so currently this file is only imports
# TODO: should be able to get rid of this file entirely

import deepspeed
import deepspeed.runtime.activation_checkpointing.checkpointing as checkpointing

# Default name for the model parallel rng tracker.
_MODEL_PARALLEL_RNG_TRACKER_NAME = deepspeed.checkpointing._MODEL_PARALLEL_RNG_TRACKER_NAME

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

