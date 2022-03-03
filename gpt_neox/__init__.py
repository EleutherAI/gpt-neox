import torch.distributed as dist


def rank_0(fn, *args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            return fn(*args, **kwargs)
    else:
        return fn(*args, **kwargs)


def print_rank_0(*args, **kwargs):
    return rank_0(print, *args, **kwargs, flush=True)
