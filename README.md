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

## Sparse Attention

To use sparse attention in your GPTNeoX model, you first need to make sure Deepspeed is installed with sparse attention enabled. You can use the following script to install all the dependencies as well as reinstall Deepspeed.

```bash
$ ./install_deepspeed.sh
```

Then

```python
model = GPTNeoX(
    num_tokens = 20000,
    dim = 512,
    seq_len = SEQ_LEN,
    depth = 12,
    heads = 8,
    sparse_attn = True,
)
```

Or if you want it for specific layers

```python
model = GPTNeoX(
    num_tokens = 20000,
    dim = 512,
    seq_len = SEQ_LEN,
    depth = 12,
    heads = 8,
    sparse_attn = (True, False) * 6, # interleaved
)
```

## Resources
If you have trouble getting the model to run, consider consulting [this guide](https://gist.github.com/kevinwatkins/232b88bfecbeca8d48d612a3e9cf65e4) to installing in a GCE virtual machine.
