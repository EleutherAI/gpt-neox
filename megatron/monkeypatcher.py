from deepspeed import comm as dist

try:
    from torch._six import inf
except ModuleNotFoundError:
    from torch import inf
import torch
from collections.abc import Iterable
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.utils import clip_tensors_by_global_norm


def get_global_norm_of_tensors(input_tensors, norm_type=2, mpu=None):
    # Overwrite of https://github.com/EleutherAI/DeeperSpeed/blob/main/deepspeed/runtime/utils.py#L866-L901
    # To support sequence parallel
    """Get norm of an iterable of tensors.
    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Taken from Nvidia Megatron.
    Arguments:
        input_tensors (Iterable[Tensor]): an iterable of Tensors will have norm computed
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
    Returns:
        Total norm of the tensors (viewed as a single vector).
    """

    assert isinstance(
        input_tensors, Iterable
    ), f"expected Iterable type not {type(input_tensors)}"
    assert all(
        [torch.is_tensor(t) for t in input_tensors]
    ), f"expected list of only tensors"

    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(t.data.abs().max() for t in input_tensors)
        total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
        if mpu is not None:
            dist.all_reduce(
                total_norm_cuda,
                op=dist.ReduceOp.MAX,
                group=mpu.get_model_parallel_group(),
            )
            dist.all_reduce(
                total_norm_cuda,
                op=dist.ReduceOp.MAX,
                group=mpu.get_seq_parallel_group(),
            )
            total_norm = total_norm_cuda[0].item()
    else:
        total_norm = sum(
            [t.data.float().norm(norm_type).item() ** norm_type for t in input_tensors]
        )
        total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
        if mpu is not None:
            dist.all_reduce(
                total_norm_cuda,
                op=dist.ReduceOp.SUM,
                group=mpu.get_model_parallel_group(),
            )
            dist.all_reduce(
                total_norm_cuda,
                op=dist.ReduceOp.SUM,
                group=mpu.get_seq_parallel_group(),
            )
        total_norm = total_norm_cuda[0].item() ** (1.0 / norm_type)

    if (
        total_norm == float("inf")
        or total_norm == -float("inf")
        or total_norm != total_norm
    ):
        total_norm = -1

    return total_norm


def replace_engine_get_global_norm(engine):
    @torch.no_grad()
    def step(closure=None):
        # to monkeypatch https://github.com/EleutherAI/DeeperSpeed/blob/main/deepspeed/runtime/bf16_optimizer.py#L233-L253
        self = engine.optimizer
        if closure is not None:
            raise NotImplementedError(f"{self.__class__} does not support closure.")

        all_groups_norm = get_global_norm_of_tensors(
            input_tensors=self.get_grads_for_norm(),
            mpu=self.mpu,
            norm_type=self.norm_type,
        )
        self._global_grad_norm = all_groups_norm

        assert all_groups_norm > 0.0
        if self.clip_grad > 0.0:
            clip_tensors_by_global_norm(
                input_tensors=self.get_grads_for_norm(for_clipping=True),
                max_norm=self.clip_grad,
                global_norm=all_groups_norm,
                mpu=self.mpu,
            )

        self.optimizer.step()

        self.update_lp_params()

        self.clear_hp_grads()

    engine.optimizer.step = step
    return engine
