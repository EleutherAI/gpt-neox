from gpt_neox.mpu_models import GPT2Model,GPT2PipelineModel
import mpu
import torch
import random
import numpy as np
#from fp16 import FP16_Module

USE_TORCH_DDP=False
if USE_TORCH_DDP:
    from torch.nn.parallel.distributed import DistributedDataParallel as DDP
#else:
#    from model import DistributedDataParallel as DDP

def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        mpu.model_parallel_cuda_manual_seed(seed)

def get_batch(context_tokens,eod_token):
    tokens = context_tokens
    batch_size = tokens.shape[0]
    tokens = tokens.view(batch_size, -1).contiguous()
    #janky for now
    seq_len = tokens.shape[1]-1

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
        tokens,
        eod_token,
        batch_size,
        seq_len,
        True,
        True)
    #TODO: Without this, mpu/transformer errors out
    attention_mask = attention_mask.half()

    return tokens, attention_mask, position_ids

def get_masks_and_position_ids(data,
                               eod_token,
                               batch_size,
                               seq_length,
                               reset_position_ids,
                               reset_attention_mask):
    #For this part, we're feeding in data that has shape[1]+1 instead of shape[1]
    data=data[:,:-1]

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i+1):, :(i+1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i+1):] -= (i + 1 - prev_index)
                    prev_index = i + 1


    return attention_mask, loss_mask, position_ids

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
    #else:
    #    model = DDP(model)

    return model