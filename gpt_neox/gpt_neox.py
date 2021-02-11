import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn, einsum
from functools import partial

from torch.utils.checkpoint import checkpoint
from einops import rearrange

from deepspeed.pipe import PipelineModule, LayerSpec

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

class MishFn(torch.autograd.Function):
    """
    Mish: A Self Regularized Non-Monotonic Neural Activation Function
    https://arxiv.org/abs/1908.08681
    """

    @staticmethod
    def forward(ctx, x):
        x_tanh_sp = torch.nn.functional.softplus(x).tanh()
        if x.requires_grad:
            ctx.save_for_backward(x_tanh_sp + x * x.sigmoid() * (1 - x_tanh_sp.square()))
        y = x * x_tanh_sp
        return y

    @staticmethod
    def backward(ctx, grad_output):
        if len(ctx.saved_tensors) == 0:
            return None
        grad, = ctx.saved_tensors
        return grad_output * grad


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
    def __init__(self, *, num_tokens, dim, seq_len, depth, heads=8, dim_head=64, attn_dropout=0., ff_dropout=0., 
                sparse_attn=False, use_fused_layernorm=False, tie_classifier_weights=False, gradient_checkpointing=True):
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
        self.depth = depth

        self.norm = norm_class(dim)

        if tie_classifier_weights:
            self.to_logits = lambda t: t @ self.token_emb.weight.t()
        else:
            self.to_logits = nn.Linear(dim, num_tokens)
        
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, x, mask=None):
        n, device = x.shape[1], x.device

        x = self.token_emb(x)
        x = self.pos_emb(torch.arange(n, device=device)) + x

        def _layer(attn, ff):
            def fn(x):
                x = attn(x) + x
                return ff(x) + x
            return fn

        if self.gradient_checkpointing:
            for (attn, ff) in self.layers:
                layer_fn = _layer(attn, ff)
                x = checkpoint(layer_fn, (x))
        else:
            for (attn, ff) in self.layers:
                layer_fn = _layer(attn, ff)
                x = layer_fn(x)

        x = self.norm(x)
        return self.to_logits(x)

class TransformerBlock(nn.Module):
    def __init__(
        self, 
        dim, 
        seq_len, 
        heads, 
        dim_head, 
        attn_dropout, 
        ff_dropout, 
        sparse_attn, 
        norm_class):
        super().__init__()

        self.attn_layer = PreNorm(dim, norm_class, Attention(dim=dim, heads=heads, seq_len=seq_len, dim_head=dim_head, dropout=attn_dropout, sparse_attn=sparse_attn))
        self.ff_layer = PreNorm(dim, norm_class, FeedForward(dim=dim, dropout=ff_dropout))

    def forward(self, input):
        x = input
        x = self.attn_layer(x) + x
        x = self.ff_layer(x) + x
        return x

class EmbedBlock(nn.Module):
    def __init__(
        self, 
        num_tokens, 
        dim, 
        seq_len):
        super().__init__()

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)

        self.token_emb.weight.data.normal_(0, 0.02)
        self.pos_emb.weight.data.normal_(0, 0.02)

    def forward(self, x):
        n, device = x.shape[1], x.device
        x = self.token_emb(x)
        x = self.pos_emb(torch.arange(n, device=device)) + x
        return x

class GPTNeoX_Pipe(PipelineModule):
    def __init__(
        self, 
        *, 
        num_tokens, 
        dim, 
        seq_len, 
        depth, 
        loss_fn,
        heads = 8, 
        dim_head = 64, 
        attn_dropout = 0., 
        ff_dropout = 0., 
        sparse_attn = False, 
        use_fused_layernorm = False, 
        tie_classifier_weights = False,
        num_stages = 2,
        **kwargs
    ):
        if not use_fused_layernorm:
            norm_class = nn.LayerNorm
        else:
            from apex.normalization import FusedLayerNorm
            norm_class = FusedLayerNorm

        self.seq_len = seq_len

        layers_sparse_attn = cast_tuple(sparse_attn, depth)

        #Build spec list
        #Input Embedding
        spec = [
            LayerSpec(EmbedBlock, num_tokens = num_tokens, dim = dim, seq_len=seq_len)
        ]
        #Transformer layers
        for i in range(depth):
            spec.append(
                LayerSpec(
                    TransformerBlock,
                    dim = dim, 
                    seq_len = seq_len, 
                    heads = heads, 
                    dim_head = dim_head, 
                    attn_dropout = attn_dropout, 
                    ff_dropout = ff_dropout, 
                    sparse_attn = layers_sparse_attn[i], 
                    norm_class = norm_class
                )
            )
        #Output norm and Linear
        spec += [
            LayerSpec(norm_class, dim),
            LayerSpec(nn.Linear, dim, num_tokens),
            lambda x: x.transpose(1, 2)
        ]
        assert len(spec) % num_stages == 0, f"for optimal performance, depth + 4 ({len(spec)}) should be divisible by the number of pipeline stages ({num_stages})"
        super().__init__(layers=spec, loss_fn=loss_fn, num_stages=num_stages, **kwargs)
