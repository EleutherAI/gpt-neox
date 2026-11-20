# LLaMA

## Training and Finetuning

These configs contain the architecture settings required to run inference/training/finetuning on the [LLaMA](https://huggingface.co/docs/transformers/main/model_doc/llama) model suite.

LLaMA finetuning  can be launched with
```sh
python ./deepy.py ./train.py -d configs llama/7B.yml llama/train_config.yml local_setup.yml
```

If training from scratch, set `finetune=False` in `./configs/llama/train_config.yml`.


## Inference


Follow the [inference guide](../../INFERENCE.md) to set the tokenizer and
checkpoint paths and create `configs/infer.yml` with `text_gen_type: input-file`.
Then run from the NeoX root, using the model config and step matching your checkpoint:
```sh
python ./deepy.py ./generate.py -d configs  \
  llama/7B.yml llama/train_config.yml local_setup.yml infer.yml \
  --iteration 10000 -i input_prompt.txt -o completions.jsonl
```
