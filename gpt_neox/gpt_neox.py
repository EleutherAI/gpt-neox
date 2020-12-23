import torch
from torch import nn, einsum
from einops import rearrange

# helpers

def exists(val):
    return val is not None

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads, causal = True, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = heads * dim_head
        self.causal = causal
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        h, device = self.heads, x.device

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if self.causal:
            i, j = sim.shape[2:]
            mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()
            mask_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(mask, mask_value)

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class GPTNeoX(nn.Module):
    def __init__(self, *, num_tokens, dim, seq_len, depth, heads = 8, dim_head = 64, attn_dropout = 0., ff_dropout = 0.):
        super().__init__()
        self.seq_len = seq_len

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)

        self.token_emb.weight.data.normal_(0, 0.02)
        self.pos_emb.weight.data.normal_(0, 0.02)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim = dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim = dim, dropout = ff_dropout)),
            ]))

        self.norm = nn.LayerNorm(dim)
        self.to_logits = lambda t: t @ self.token_emb.weight.t()

    def forward(self, x, mask = None):
        n, device = x.shape[1], x.device

        x = self.token_emb(x)
        x = self.pos_emb(torch.arange(n, device = device)) + x

        for (attn, ff) in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        x = self.norm(x)
        return self.to_logits(x)
