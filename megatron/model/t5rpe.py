# Based on: https://github.com/lucidrains/x-transformers/blob/6b93c21be0d0a679da6f7b9621d9bb638ab18428/x_transformers/x_transformers.py#L106 (14.12.2021)

import math
import torch
from torch import nn
from einops import rearrange


class RelativePositionBias(nn.Module):
    def __init__(self, causal = False, num_buckets = 32, max_distance = 128, heads = 8):
        super().__init__()
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal = True, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, q_len, k_len):
        q_pos = torch.arange(q_len, dtype = torch.long, device = torch.cuda.current_device())
        k_pos = torch.arange(k_len, dtype = torch.long, device = torch.cuda.current_device())
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j h -> () h i j')
        return bias

