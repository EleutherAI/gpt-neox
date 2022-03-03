import torch.nn as nn


def gpt2_attention_mask_func(attention_scores, ltor_mask):
    attention_scores.masked_fill_(ltor_mask, -10000.0)
    return attention_scores

class GPT2Model(nn.Module):
    pass