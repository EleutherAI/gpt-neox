import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

path=Path(__file__).resolve().parents[2]/'megatron/bf16_accumulation.py'
spec=importlib.util.spec_from_file_location('bf16_accumulation',path)
accumulation=importlib.util.module_from_spec(spec)
spec.loader.exec_module(accumulation)


@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float32])
@pytest.mark.parametrize('clear',[False,True])
def test_cast_and_add_matches_explicit_fp32_conversion_bitwise(dtype,clear):
    torch.manual_seed(12)
    actual=torch.randn(128,128)*torch.logspace(-8,8,128)
    expected=actual.clone()
    optimizer=SimpleNamespace(fp32_groups_gradients=[[actual]],fp32_groups_has_gradients=[[False]])
    parameter=torch.nn.Parameter(torch.zeros_like(actual,dtype=dtype))
    for _ in range(16):
        gradient=(torch.randn_like(actual)*torch.logspace(-8,8,128)).to(dtype)
        parameter.grad=gradient.clone()
        expected.add_(gradient.to(torch.float32))
        accumulation.cast_add_gradient(optimizer,parameter,0,0,clear)
        assert torch.equal(actual,expected)
        assert optimizer.fp32_groups_has_gradients[0][0]
        assert parameter._hp_grad is actual
        assert torch.equal(parameter.grad,torch.zeros_like(gradient)if clear else gradient)
    parameter.grad=None
    accumulation.cast_add_gradient(optimizer,parameter,0,0,clear)
    assert torch.equal(actual,expected)
