import torch
from torch import nn
import torch.nn.functional as F

# nucleus

def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

# topk

def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

class AutoregressiveWrapper(nn.Module):
    def __init__(self, net, ignore_index = 0, pad_value = 0):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        self.seq_len = net.seq_len

    @torch.no_grad()
    def generate(self, start_tokens, seq_len, eos_token = None, temperature = 1., filter_logits_fn = top_k, filter_thres = 0.9, **kwargs):
        device = start_tokens.device
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        out = start_tokens
        mask = kwargs.pop('mask', None)

        if mask is None:
            mask = torch.full_like(out, True, dtype=torch.bool, device=out.device)

        for _ in range(seq_len):
            x = out[:, -self.seq_len:]
            mask = mask[:, -self.seq_len:]

            logits = self.net(x, mask=mask, **kwargs)[:, -1, :]
            filtered_logits = filter_logits_fn(logits, thres = filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)
            mask = F.pad(mask, (0, 1), value=True)

            if eos_token is not None and (sample == eos_token).all():
                break

        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)
        return out

    def forward(self, x, **kwargs):
        xi = x[:, :-1]
        xo = x[:, 1:]

        # help auto-solve a frequent area of confusion around input masks in auto-regressive
        # if user supplies a mask that is only off by one from the source sequence, resolve it for them
        mask = kwargs.pop('mask', None)
        if mask is not None and mask.shape[1] == x.shape[1]:
            mask = mask[:, :-1]
            kwargs.update(mask = mask)

        out = self.net(xi, **kwargs)

        losses = F.cross_entropy(out.transpose(1, 2), xo, reduction='none', ignore_index = self.ignore_index)
        loss = losses.mean()
        
        return loss
