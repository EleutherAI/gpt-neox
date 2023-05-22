import mii
from transformers import AutoConfig

mii_config = {"dtype": "fp16"}

name = "EleutherAI/pythia-160m"

config = AutoConfig.from_pretrained(name)
model_hidden_size = config.hidden_size

ds_config = {
    "fp16": {
        "enabled": True
    },
    "bf16": {
        "enabled": False
    },
    "aio": {
        "block_size": 262144,
        "queue_depth": 32,
        "thread_count": 1,
        "single_submit": False,
        "overlap_events": True
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": model_hidden_size * model_hidden_size,
        "stage3_prefetch_bucket_size": 0.1 * model_hidden_size * model_hidden_size,
        "stage3_max_live_parameters": 1e8,
        "stage3_max_reuse_distance": 1e8,
        "stage3_param_persistence_threshold": 10 * model_hidden_size
    },
    "train_micro_batch_size_per_gpu": 1,
}

mii.deploy(task='text-generation',
           model=name,
           deployment_name=name + "_deploy",
           mii_config=mii_config,
           enable_deepspeed=False,
           enable_zero=True,
           ds_config=ds_config)