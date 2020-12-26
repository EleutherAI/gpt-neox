# GPT-NeoX
An implementation of model parallel GPT-3-like models on GPUs, based on the DeepSpeed library. Designed to be able to train models in the hundreds of billions of parameters or larger.

## Requirements

```bash
$ pip install -r requirements.txt
```

Test deepspeed locally

```bash
$ deepspeed train_enwik8.py \
	--deepspeed \
	--deepspeed_config ./configs/base_deepspeed.json
```
