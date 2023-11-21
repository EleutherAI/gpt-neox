import pytest
from megatron import fused_kernels


def test_fused_kernels():
    pytest.xfail(reason="Fused kernels require manual intervention to install")
    fused_kernels.load_fused_kernels()
