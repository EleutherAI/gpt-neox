# GPT-NeoX
An implementation of model parallel GPT-3-like models on GPUs, based on the DeepSpeed library. Designed to be able to train models in the hundreds of billions of parameters or larger. This repository is under development and may change rapidly without warning.

## Requirements

Everything you need to get started running the code can be installed via pip:
```bash
$ pip install -r requirements.txt
```
**Important: This codebase does not install Microsoft's DeepSpeed library.** It installs [DeeperSpeed](www.GitHub.com/eleutherai/DeeperSpeed), EleutherAI's variant on the original [DeepSpeed](www.GitHub.com/Microsoft/DeepSpeed). We have added some necessary functionality for our purposes and patched holes created by the fact that only parts of DeepSpeed were publicly released, but DeeperSpeed uses the same namespace as DeepSpeed and may break other code built upon DeepSpeed. **If you use or suspect you might use Microsoft's DeepSpeed for another project**, we strongly secommend you use `anaconda` to install this code in an isolated environment by creating a condo environment and running `conda install --file requirements.txt`. We welcome any suggestions for improvements to our DeepSpeeder library, but please open issues on [its repo](www.GitHub.com/eleutherai/DeeperSpeed) rather than this one. 

EleutherAI members who wish to run models on our Kubernetes cluster will additionally need to install Kubernetes and obtain an authorization from Stella Biderman or Sid Black. Please reach out on discord in the #gpt-neo channel. You will also need to create a [WandB](https://wandb.ai/home) account and share your username so that you can be added to the organization WandB account.

## Running the code

The core anatomy of a call to the DeepSpeed engine is the following
```bash
$ deepspeed --hostfile=host_path train_script.py user_args\
	--deepspeed \
	--deepspeed_config deepspeed_config.json
```
where
- `host_path` (optional) is the path to the host file containing the addresses of the machines you wish to train on.
- `train_script.py` is the training script you wish to use. Our main training script is `train_pipeline.py`.
- `deepspeed_config.json` is the `json` file containing DeepSpeed-specific hyperparameters.

In this repository, we provide a lightweight wrapper for the above function call for two main reasons. Firstly, we find the way the arguments are ordered and used somewhat counterintuitive, and secondly our wrapper automatically uploads logging data to WandB. Everything in this repository will work with both the native `DeepSpeed` command and with our `deepy` command. The core anatomy of a `deepy` call is
```bash
$ ./deepy --hostfile=host_path train_script.py deepspeed_config.json
```

### Running the code locally
This code is set up to run automatically on as many GPUs as are avaliable. If you have multiple GPUs and only wish to make use of some of them, you can find information about how to specify which GPU(s) to use in training [here](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node).

The most common pitfall for local training is pipeline parallelism. Pipeline parallelism paritions the model into segments (called `PipelineModule`s in our code) that can decrese latency by running partially asynchronously.

### Running the code on a server

This code is set up to run automatically on as many GPUs as are avaliable. To run across multiple machines, you need to make use of a hostfile which lists the IP address of each machine you wish to run the code on followed by the number of GPUs to use. For example, `123.45.67.890 slots=8` instructs the code to run on all eight GPUs of the machine at `123.45.67.890`. Each machine should be listed on a separate line with no end-of-line punctuation. It is officially recommended that you set up passwordless ssh, but we have had success entering the password at run-time. To have your hostfile used by GPT-NeoX automatically, store it at `~/jobs/hostfile`. Otherwise, you can provide it as an argument as shown above.

**EleutherAI members:** Once you have been granted access to the EleutherAI servers and have confirmed that an unused cluster is currently running, simply ssh into the cluster. If you have been granted the ability to create an destroy Kubernetes clusters, run `kubernetes/deploy_k8s.sh branch_name num_pods cluster_name` to create a cluster.

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
