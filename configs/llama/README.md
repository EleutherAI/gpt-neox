# LLaMA

## Training and Finetuning

These configs contain the architecture settings required to run inference/training/finetuning on the [LLaMA](https://huggingface.co/docs/transformers/main/model_doc/llama) model suite.

LLaMA finetuning  can be launched with
```sh
python ./deepy.py ./train.py -d configs llama/7B.yml llama/train_config.yml local_setup.yml
```

If training from scratch, set `finetune=False` in `./configs/llama/train_config.yml`.


## Inference


LLaMA generation can be launched with
```sh
python ./deepy.py ./generate.py -d configs  \
  llama/7B.yml llama/train_config.yml local_setup.yml text_generation.yml \
  -i input_prompt.txt -o prompt_out.txt
```
