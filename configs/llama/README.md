# Llama
These configs contain the architecture settings required to run inference/training/finetuning on the [Llama](https://huggingface.co/docs/transformers/main/model_doc/llama) model suite.

Llama finetuning  can be launched with
```sh
python ./deepy.py ./train.py ./configs/llama/7B.yml ./configs/llama/train_config.yml ./configs/local_setup.yml
```

If training from scratch, set `finetune=False` in `./configs/llama/train_config.yml`
