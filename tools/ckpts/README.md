# Checkpoint Scripts


## Utilities

### `inspect_checkpoints.py`
Reports information about a saved checkpoint.
```
usage: inspect_checkpoints.py [-h] [--attributes [ATTRIBUTES ...]] [--interactive] [--compare] [--diff] dir

positional arguments:
  dir                   The checkpoint dir to inspect. Must be either: - a directory containing pickle binaries saved with 'torch.save' ending in .pt or .ckpt - a single path to a .pt or .ckpt file - two comma separated directories -
                        in which case the script will *compare* the two checkpoints

options:
  -h, --help            show this help message and exit
  --attributes [ATTRIBUTES ...]
                        Name of one or several attributes to query. To access an attribute within a nested structure, use '/' as separator.
  --interactive, -i     Drops into interactive shell after printing the summary.
  --compare, -c         If true, script will compare two directories separated by commas
  --diff, -d            In compare mode, only print diffs
```

## HuggingFace Scripts

### `convert_hf_to_sequential.py`
A script for converting publicly available Huggingface (HF) checkpoints to NeoX format.

Note that this script requires access to corresponding config files for equivalent NeoX models to those found in Hugging face.

```
Example usage: (Converts the 70M Pythia model to NeoX format)
================================================================
OMPI_COMM_WORLD_RANK=0 CUDA_VISIBLE_DEVICES=0 python tools/ckpts/convert_hf_to_sequential.py \
    --hf-model-name pythia-70m-v0 \
    --revision 143000 \
    --output-dir checkpoints/neox_converted/pythia/70m \
    --cache-dir checkpoints/HF \
    --config configs/pythia/70M.yml configs/local_setup.yml \
    --test


For multi-gpu support we must initialize deepspeed:
NOTE: This requires manually changing the arguments below.
================================================================
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./deepy.py tools/ckpts/convert_hf_to_sequential.py \
    -d configs pythia/70M.yml local_setup.yml
```
### `convert_module_to_hf.py`
Converts a NeoX model with pipeline parallelism greater than 1 to a HuggingFace transformers `GPTNeoXForCausalLM` model

Note that this script does not support all NeoX features.
Please investigate carefully whether your model is compatible with all architectures supported by the GPTNeoXForCausalLM class in HF.

(e.g. position embeddings such as AliBi may not be supported by Huggingface's GPT-NeoX architecture)

```
usage: convert_module_to_hf.py [-h] [--input_dir INPUT_DIR] [--config_file CONFIG_FILE] [--output_dir OUTPUT_DIR] [--upload]

Merge MP partitions and convert to HF Model.

options:
  -h, --help            show this help message and exit
  --input_dir INPUT_DIR
                        Path to NeoX checkpoint, e.g. /path/to/model/global_step143000
  --config_file CONFIG_FILE
                        Path to config file for the input NeoX checkpoint.
  --output_dir OUTPUT_DIR
                        Output dir, where to save the HF Model, tokenizer, and configs
  --upload              Set to true in order to upload to the HF Hub directly.
```

### `convert_sequential_to_hf.py`
Converts a NeoX model without pipeline parallelism to a HuggingFace transformers `GPTNeoXForCausalLM` model.

```
usage: convert_sequential_to_hf.py [-h] [--input_dir INPUT_DIR] [--config_file CONFIG_FILE] [--output_dir OUTPUT_DIR] [--upload]

Merge MP partitions and convert to HF Model.

options:
  -h, --help            show this help message and exit
  --input_dir INPUT_DIR
                        Path to NeoX checkpoint, e.g. /path/to/model/global_step143000
  --config_file CONFIG_FILE
                        Path to config file for the input NeoX checkpoint.
  --output_dir OUTPUT_DIR
                        Output dir, where to save the HF Model, tokenizer, and configs
  --upload              Set to true in order to upload to the HF Hub directly.
```
### `upload.py`
Uploads a _converted_ checkpoint to the HuggingFace hub.

```
python upload.py <converted-ckpt-dir> <repo-name> <branch-name>
```
## NeoX-20B Scripts

### `merge20b.py`
Reduces model and pipeline parallelism of a 20B checkpoint to 1 and 1.

```
usage: merge20b.py [-h] [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR]

Merge 20B checkpoint.

options:
  -h, --help            show this help message and exit
  --input_dir INPUT_DIR
                        Checkpoint dir, which should contain (e.g. a folder named "global_step150000")
  --output_dir OUTPUT_DIR
                        Output dir, to save the 1-GPU weights configs
```
## Llama Scripts

### `convert_raw_llama_weights_to_neox.py`
Takes a Llama checkpoint and puts it into a NeoX-compatible format.

```
usage: convert_raw_llama_weights_to_neox.py [-h] [--input_dir INPUT_DIR] [--model_size {7B,13B,30B,65B,tokenizer_only}] [--output_dir OUTPUT_DIR] [--num_output_shards NUM_OUTPUT_SHARDS] [--pipeline_parallel]

Convert raw LLaMA checkpoints to GPT-NeoX format.

options:
  -h, --help            show this help message and exit
  --input_dir INPUT_DIR
                        Location of LLaMA weights, which contains tokenizer.model and model folders
  --model_size {7B,13B,30B,65B,tokenizer_only}
  --output_dir OUTPUT_DIR
                        Location to write GPT-NeoX mode
  --num_output_shards NUM_OUTPUT_SHARDS
  --pipeline_parallel   Only use if PP>1
```
