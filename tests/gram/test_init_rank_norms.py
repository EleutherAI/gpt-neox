"""Adapter init scale, per-rank routing draws and per-role gradient norms."""
import pytest
import torch
from torch import nn

from test_routing import Network, args, gram, setup


class Core(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.linear(x), None


def test_adapter_output_projection_uses_scaled_init():
    torch.manual_seed(0)
    mlp = gram.AuxiliaryMLP(Core(512), 512, 2048, gram.RoutingController(), init_std=0.02, num_layers=32)
    assert mlp.gram_aux[0].weight.std().item() == pytest.approx(0.02, rel=0.02)
    assert mlp.gram_aux[2].weight.std().item() == pytest.approx(0.02 / 64 ** 0.5, rel=0.02)
    assert mlp.gram_aux[0].bias.abs().sum() == 0 and mlp.gram_aux[2].bias.abs().sum() == 0


def test_adapter_uses_the_model_init_functions_when_given():
    calls = []

    def first(w):
        calls.append("in")
        return nn.init.constant_(w, 1.0)

    def second(w):
        calls.append("out")
        return nn.init.constant_(w, 2.0)

    mlp = gram.AuxiliaryMLP(Core(8), 8, 4, gram.RoutingController(nested=True), init_std=0.02,
                            init_method=first, output_layer_init_method=second)
    assert calls == ["in", "out", "in", "out"]
    for branch in (mlp.gram_aux, mlp.gram_inner):
        assert torch.all(branch[0].weight == 1.0) and torch.all(branch[2].weight == 2.0)


def test_install_scales_output_init_by_num_layers():
    torch.manual_seed(0)
    model = Network()
    gram.install(model, args(gram_aux_width=4096, num_layers=2, init_method_std=0.02))
    w = model.layers[0].mlp.gram_aux[2].weight
    assert w.std().item() == pytest.approx(0.02 / 4 ** 0.5, rel=0.05)
    assert model.layers[0].mlp.gram_aux[0].weight.std().item() == pytest.approx(0.02, rel=0.05)


def draws(rank, n=400):
    controller = gram.RoutingController()
    controller.rank = rank
    out = []
    for micro in range(n):
        controller.select(0, 0.5, 1234, 7, micro)
        out.append(controller.aux_on)
    return out


def test_routing_draws_depend_on_data_parallel_rank():
    a, b = draws(0), draws(1)
    assert a == draws(0)  # reproducible for a rank
    assert a != b  # independent across ranks
    assert 0.4 < sum(a) / len(a) < 0.6 and 0.4 < sum(b) / len(b) < 0.6
    agree = sum(x == y for x, y in zip(a, b)) / len(a)
    assert 0.4 < agree < 0.6


def test_optimizer_step_records_per_role_gradient_norms():
    model, controller, opt = setup()
    controller.select(0, 0.0, 1234, 3, 0)
    model(torch.tensor([[1, 2, 3, 4]])).square().mean().backward()
    expected = [0.0] * controller.num_roles
    for group in opt.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                expected[group["gram_role"]] += p.grad.float().pow(2).sum().item()
    opt.step()
    assert controller.role_grad_sq.tolist() == pytest.approx(expected)
    assert expected[0] > 0 and expected[1] == 0  # clean data, no robustness: adapters untouched


def test_record_grad_norms_undoes_clipping_and_tracks_interval_max():
    controller = gram.RoutingController()
    controller.role_grad_sq = torch.tensor([1.0, 0.25])
    controller.record_grad_norms(4.0, 1.0)  # clipped by 1/4
    controller.role_grad_sq = torch.tensor([0.04, 0.0])
    controller.record_grad_norms(0.5, 1.0)  # not clipped
    log = controller.pop_grad_norm_log()
    assert log["grad_norm"] == 0.5 and log["grad_norm_max"] == 4.0
    assert log["grad_norm_role0"] == pytest.approx(0.2) and log["grad_norm_role0_max"] == pytest.approx(4.0, rel=1e-5)
    assert log["grad_norm_role1_max"] == pytest.approx(2.0, rel=1e-5)
    assert controller.pop_grad_norm_log() == {}
    controller.record_grad_norms(None, 1.0)
    assert controller.pop_grad_norm_log() == {}
