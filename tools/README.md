# NeoX Utility Scripts and Tools

## Checkpoint Conversion

The default format Deepspeed checkpoints are saved in is dependent on the model and pipeline parallelism settings of the training run. Running a model on a cluster with a different number or type of GPUs is difficult. We have adapted a set of scripts developed by [BigScience](https://github.com/bigscience-workshop/Megatron-DeepSpeed/tree/main/tools/convert_checkpoint) to make this easier.

### DeeperSpeed to universal

To convert your checkpoint to the universal checkpoint format run the `ds_to_universal.py` script with a command along these lines.

```bash
CURR_CKPT="/path/to/your/old/checkpoint"
NEW_CKPT="/path/where/you/want/the/new/checkpoint"
CFG="/path/to/model/config/file"

python3 tools/ds_to_universal.py \
    --input_folder $CURR_CKPT \
    --output_folder $NEW_CKPT \
    --config $CFG
```

To then run the model from your new checkpoint, add these lines to a new config and run your model like you normally would.

```json
{
    "load": "/path/where/you/want/the/new/checkpoint",
    "load_universal": true
}
```

### DeeperSpeed to DeeperSpeed Reshaping

To reshape a DeeperSpeed checkpoint to _reduce_ the parallelism settings, you can use the `deepspeed_to_deepspeed.py` script. It does not work if you would like to re-shard a model to increase the amount of tensor or pipeline parallelism. But if you would like to decrease the amount of parallelism you can run the script with a command like the one below.

```bash
CURR_CKPT="/path/to/your/old/checkpoint"
NEW_CKPT="/path/where/you/want/the/new/checkpoint"
CFG="/path/to/model/config/file"
TP=1 # Tensor (model) parallelism setting for the new checkpoint, must be less than or equal to the model's original tensor parallelism
DP=1 # Data parallelism setting for the new checkpoint
PP=1 # Model parallelism setting for the new checkpoint, must be less than or equal to the model's original pipeline parallelism

python3 tools/deepspeed_to_deepspeed.py \
    --input_folder $CURR_CKPT \
    --output_folder $NEW_CKPT \
    --config $CFG \
    --target_tp $TP \
    --target_dp $DP \
    --target_pp $PP
```
