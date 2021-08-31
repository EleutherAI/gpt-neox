import torch
import math 

class SinusoidalPositionalEmbedding(torch.nn.Module):

    def __init__(self, dim, base=10000, precision=torch.half):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.precision = precision

    def forward(self, x, seq_dim=1):
        t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum("i,j->ij", t, self.inv_freq)
        if self.precision == torch.bfloat16:
            sinusoid_inp = sinusoid_inp.float()
        sin, cos = sinusoid_inp.sin(), sinusoid_inp.cos()
        if self.precision == torch.bfloat16:
            sin, cos = sin.bfloat16(), cos.bfloat16()
        emb = torch.cat((sin, cos), dim=-1)
        return emb[None, :, :]


class RotaryEmbedding(torch.nn.Module):
    
    def __init__(self, dim, base=10000, precision=torch.half):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.precision = precision

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]
            if self.precision == torch.bfloat16:
                self.cos_cached = self.cos_cached.bfloat16()
                self.sin_cached = self.sin_cached.bfloat16()
        return self.cos_cached, self.sin_cached


# rotary pos emb helpers:

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1) # dim=-1 triggers a bug in earlier torch versions

@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos, sin = cos[offset:q.shape[0]+offset, ...], sin[offset:q.shape[0]+offset, ...]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

def apply_rotary_pos_emb_torch(q, k, cos, sin, offset: int = 0): # jitting fails with bf16
    cos, sin = cos[offset:q.shape[0]+offset, ...], sin[offset:q.shape[0]+offset, ...]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class AliBi(torch.nn.Module):

  def __init__(self, num_heads, mp_size=1, mp_rank=1):
    super().__init__()
    # megatron splits across heads, so we need to make sure each
    # head receives the correct matrix
    assert mp_size <= num_heads and mp_rank <= mp_size
    self.mp_size = mp_size
    self.mp_rank = mp_rank
    self.num_heads = num_heads
    self.slice_size = num_heads // mp_size
    self.cached_matrix = None
    self.cached_seq_len = None
    slopes = torch.Tensor(self._get_slopes(num_heads))[mp_rank * self.slice_size : (mp_rank + 1)  * self.slice_size]
    self.register_buffer('slopes', slopes)

  
  def _get_slopes(self, n):
    """
    Get slopes for Alibi positional embedding
    n : int = number of heads. 
    For best performance, restrict n to a power of 2.
    """
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)                   
    else:                                                 
        closest_power_of_2 = 2**math.floor(math.log2(n)) 
        return get_slopes_power_of_2(closest_power_of_2) + self._get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]

  def forward(self, x):
    # [b, np, sq, sk]
    seq_len = x.shape[-1]
    if self.cached_seq_len != seq_len:
        a = -torch.tril(torch.arange(seq_len).view(seq_len, 1).repeat(1, seq_len) + torch.arange(0, -seq_len, -1))
        a = a.to(x.device).to(x.dtype)
        slopes = self.slopes.to(a.device).to(a.dtype)
        a = a * slopes.view(self.slopes.shape[0], 1, 1)
        self.cached_seq_len = seq_len
        self.cached_matrix = a
    else:
      a = self.cached_matrix
    return x + a