from gpt_neox.mpu_models import GPT2Model,GPT2PipelineModel
import mpu
import torch
#from fp16 import FP16_Module

USE_TORCH_DDP=True
if USE_TORCH_DDP:
    from torch.nn.parallel.distributed import DistributedDataParallel as DDP
else:
    from model import DistributedDataParallel as DDP

def get_model(vocab_size,params):
    """Build the model."""

    print('building GPT2 model ...')
    #TODO: Change the dropout probs, this is just using defaults from megatron code
    model = GPT2Model(num_layers=params['n_layers'],
                      vocab_size=vocab_size,
                      hidden_size=params['hidden_dim'],
                      num_attention_heads=params['n_heads'],
                      embedding_dropout_prob=params.get("hidden_dropout", 0.1),
                      attention_dropout_prob=params.get("attention_dropout", 0.1),
                      output_dropout_prob=params.get("hidden_dropout", 0.1),
                      max_sequence_length=params['seq_len'],
                      checkpoint_activations=params.get("gradient_checkpointing", True),
                      checkpoint_num_layers=params.get("checkpoint_num_layers", 1),
                      parallel_output=True)

    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    model.half()

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    #model = FP16_Module(model)

    # Wrap model for distributed training.
    if USE_TORCH_DDP:
        i = torch.cuda.current_device()
        model = DDP(model, device_ids=[i], output_device=i,
                    process_group=mpu.get_data_parallel_group())
    else:
        model = DDP(model)

    return model