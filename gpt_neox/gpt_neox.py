import torch
import torch.nn.functional as F
from torch import nn, einsum
from functools import partial

from einops import rearrange

# helpers

def exists(val):
    return val is not None

def cast_tuple(val, depth):
    if isinstance(val, tuple):
        return val
    return (val,) * depth

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, norm_class, fn):
        super().__init__()
        self.fn = fn
        self.norm = norm_class(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)


# attention

def dense_attn(q, k, v, attn_mask = None, dropout_fn = None):
    scale = q.shape[-1] ** -0.5
    sim = einsum('b h i d, b h j d -> b h i j', q, k) * scale

    if exists(attn_mask):
        sim = sim + attn_mask[None, None, :, :]

    attn = sim.softmax(dim=-1)

    if exists(dropout_fn):
        attn = dropout_fn(attn)

    out = einsum('b h i j, b h j d -> b h i d', attn, v)
    return out

class Attention(nn.Module):
    def __init__(self, dim, heads, seq_len, causal=True, dim_head=64, dropout=0., sparse_attn=False):
        super().__init__()
        inner_dim = heads * dim_head
        self.causal = causal
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dropout = nn.Dropout(dropout)

        if sparse_attn:
            from deepspeed.ops.sparse_attention import SparseSelfAttention, VariableSparsityConfig

            sparsity_config = VariableSparsityConfig(
                num_heads=heads,
                attention=("unidirectional" if causal else "bidirectional")
            )

            self.attn_fn = SparseSelfAttention(
                sparsity_config=sparsity_config,
                max_seq_length=seq_len,
                attn_mask_mode='add'
            )
        else:
            self.attn_fn = partial(dense_attn, dropout_fn = self.dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, **kwargs):
        b, h, device = x.shape[0], self.heads, x.device

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        mask = None
        if self.causal:
            i, j = q.shape[-2], k.shape[-2]
            bool_mask = torch.ones(i, j, device=device).triu_(j - i + 1).bool()
            mask = torch.zeros(i, j, device=device).to(q)
            mask_value = -(torch.finfo(q.dtype).max / 2)
            mask.masked_fill_(bool_mask, mask_value)

        out = self.attn_fn(q, k, v, attn_mask=mask)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class GPTNeoX(nn.Module):
    def __init__(self, *, num_tokens, dim, seq_len, depth, heads=8, dim_head=64, attn_dropout=0., ff_dropout=0., sparse_attn=False, use_fused_layernorm=False, tie_classifier_weights=False):
        super().__init__()
        if not use_fused_layernorm:
            norm_class = nn.LayerNorm
        else:
            from apex.normalization import FusedLayerNorm
            norm_class = FusedLayerNorm

        self.seq_len = seq_len

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)

        self.token_emb.weight.data.normal_(0, 0.02)
        self.pos_emb.weight.data.normal_(0, 0.02)

        self.layers = nn.ModuleList([])
        layers_sparse_attn = cast_tuple(sparse_attn, depth)

        for _, layer_sparse_attn in zip(range(depth), layers_sparse_attn):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, norm_class, Attention(dim=dim, heads=heads, seq_len=seq_len, dim_head=dim_head, dropout=attn_dropout, sparse_attn=layer_sparse_attn)),
                PreNorm(dim, norm_class, FeedForward(dim=dim, dropout=ff_dropout)),
            ]))

        self.norm = norm_class(dim)

        if tie_classifier_weights:
            self.to_logits = lambda t: t @ self.token_emb.weight.t()
        else:
            self.to_logits = nn.Linear(dim, num_tokens)

    def forward(self, x, mask=None):
        n, device = x.shape[1], x.device

        x = self.token_emb(x)
        x = self.pos_emb(torch.arange(n, device=device)) + x

        for (attn, ff) in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        x = self.norm(x)
        return self.to_logits(x)
