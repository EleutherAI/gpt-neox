import os
from pathlib import Path

from oslo.pytorch._C import Binder


scaled_upper_triang_masked_softmax_cuda = None


class FusedSoftmaxBinder(Binder):
    @property
    def name(self):
        return "neox_softmax"

    @property
    def base_path(self):
        from gpt_neox import csrc

        return Path(csrc.__file__).parent.absolute()

    def includes(self):
        return [os.path.join(self.base_path, "includes")]

    def sources(self):
        return [
            "scaled_upper_triang_masked_softmax.cpp",
            "scaled_upper_triang_masked_softmax_cuda.cu",
        ]


def get_scaled_upper_triang_masked_softmax_cuda():
    global scaled_upper_triang_masked_softmax_cuda

    if scaled_upper_triang_masked_softmax_cuda is None:
        scaled_upper_triang_masked_softmax_cuda = FusedSoftmaxBinder().bind()

    return scaled_upper_triang_masked_softmax_cuda
