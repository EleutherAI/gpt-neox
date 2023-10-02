# Checkpoint Scripts


# Utilities

* `inspect_checkpoints.py` reports information about a saved checkpoint.
* `merge_mp_partitions.py` reduce model (aka tensor) parallelism of a saved checkpoint.


## HuggingFace Scripts

* `convert_hf_to_sequential.py` converts a HuggingFace model to a NeoX compatible format
* `convert_module_to_hf.py` converts a NeoX model to a HuggingFace transformers `GPTNeoXForCausalLM` model
* `convert_sequential_to_hf.py` converts a NeoX model to a HuggingFace transformers `GPTNeoXForCausalLM` model.
* `upload.py` uploads a _converted_ checkpoint to the HuggingFace hub.


## NeoX-20B Scripts

* `merge20b.py` reduces model and pipeline parallelism of a 20B checkpoint to 1 and 1.

## Llama Scripts

* `convert_raw_llama_weights_to_neox.py` takes a Llama checkpoint and puts it into a NeoX-compatible format.
