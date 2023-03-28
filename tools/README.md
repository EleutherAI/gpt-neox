# Checkpoint Conversion

The default format Deepspeed checkpoints are saved in is dependent on the model and pipeline parallelism settings of the training run. Running a model on a cluster with a different number or type of GPUs is difficult. We have adapted a set of scripts developed by [BigScience](https://github.com/bigscience-workshop/Megatron-DeepSpeed/tree/main/tools/convert_checkpoint) to make this easier.

## Deepspeed to universal

```bash
CURR_CKPT="/path/to/your/old/checkpoint"

python3 tools/ds_to_universal.py --input_
```

## Deeperspeed to Deeperspeed Reshaping
