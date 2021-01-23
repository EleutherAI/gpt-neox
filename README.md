# GPT-NeoX
An implementation of model parallel GPT-3-like models on GPUs, based on the DeepSpeed library. Designed to be able to train models in the hundreds of billions of parameters or larger. This repository is under development and may change rapidly without warning.

## Requirements

```bash
$ pip install -r requirements.txt
```

## Running the code

The anatomy of a call to the DeepSpeed engine is the following
```bash
$ deepspeed --hostfile=host_path train_script.py \
	--deepspeed \
	--deepspeed_config ./configs/base_deepspeed.json
```

### Running the code locally

### Running the code on a server

This code is set up to run automatically on as many GPUs as are avaliable. To run across multiple machines, you need to make use of a hostfile which lists the IP address of each machine you wish to run the code on followed by the number of GPUs to use. For example, `123.45.67.890 slots=8` instructs the code to run on all eight GPUs of the machine at `123.45.67.890`. Each machine should be listed on a separate line with no end-of-line punctuation. It is officially recommended that you set up passwordless ssh, but we have had success entering the password at run-time. To have your hostfile used by GPT-NeoX automatically, store it at `~/jobs/hostfile`. Otherwise, you can provide it as an argument as shown above.

**EleutherAI members:**

### ~/scripts/

The directory `~/scripts/` stores various scripts for automatically starting runs with particular settings and configs that we have found useful. They can be run using `sh scripts/script_name.sh` but should not be relied upon. We do not guarentee forward compatibility of any scripts.

## Datasets

### Tokenizers

### Using our data

### Using your data

## Advanced Options

## Contribute

If you want to get involved, check out our repo projects. Anything that is listed as "todo" or has not been assigned to anyone is fair game, but please leave a comment so that we know you're working on it!

## Resources
If you have trouble getting the model to run, consider consulting [this guide](https://gist.github.com/kevinwatkins/232b88bfecbeca8d48d612a3e9cf65e4) to installing in a GCE virtual machine. You may also find the (very sparse) [DeepSpeed docs](https://www.deepspeed.ai) helpful.
