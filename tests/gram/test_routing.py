"""CPU checks of the module imported by the actual NeoX trainer."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

SOURCE = Path(__file__).resolve().parents[2] / "megatron/gram.py"
spec = importlib.util.spec_from_file_location("neox_gram", SOURCE)
gram = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gram)


class ParallelMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, x):
        return self.linear(x), None


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Linear(8, 8)
        self.norm = nn.LayerNorm(8)
        self.mlp = ParallelMLP()

    def forward(self, x):
        return x + self.mlp(self.norm(self.attention(x)))[0]


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.layers = nn.Sequential(Block(), Block())
        self.head = nn.Linear(8, 16)

    def forward(self, x):
        return self.head(self.layers(self.embed(x)))


def args(**kw):
    values = dict(is_pipe_parallel=False, model_parallel_size=1, precision="fp32",
                  zero_optimization={"stage": 0}, te_layernorm_mlp=False, te_mha=False,
                  moe_num_experts=1, train_impl="normal", train_micro_batch_size_per_gpu=1,
                  gram_labels_path="labels.npy", use_mup=False, use_bnb_optimizer=False,
                  optimizer_type="adam", gram_aux_width=4, gram_core_robustness=0.5,
                  num_layers=2, hidden_size=8, init_method_std=0.02)
    values["optimizer"] = {"params": {"lr": 0.01}}
    return SimpleNamespace(**(values | kw))


def setup():
    torch.manual_seed(7)
    model = Network()
    controller = gram.install(model, args())
    opt = gram.RoutedAdamW(model.parameters(), controller, lr=0.01, weight_decay=0.1)
    return model, controller, opt


def backward(model, controller, flagged, micro=0, robust=0):
    controller.select(flagged, robust, 1234, 3, micro)
    model(torch.tensor([[1, 2, 3, 4]])).square().mean().backward()


def test_flagged_freezes_every_core_parameter_but_trains_early_auxiliary():
    model, controller, opt = setup()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    backward(model, controller, True)
    assert model.layers[0].mlp.gram_aux[0].weight.grad.abs().sum() > 0
    opt.step()
    for name, p in model.named_parameters():
        if p.gram_role == 0:
            torch.testing.assert_close(p, before[name], rtol=0, atol=0)
            assert p not in opt.state
    assert any(not torch.equal(p, before[n]) for n, p in model.named_parameters() if p.gram_role)


def test_core_only_skips_aux_momentum_decay_and_step_count():
    model, controller, opt = setup()
    backward(model, controller, True)
    opt.step()
    opt.zero_grad(set_to_none=True)
    aux = model.layers[0].mlp.gram_aux[0].weight
    before = aux.detach().clone()
    state = copy.deepcopy(opt.state[aux])
    backward(model, controller, False)
    opt.step()
    torch.testing.assert_close(aux, before, rtol=0, atol=0)
    for k in state:
        torch.testing.assert_close(opt.state[aux][k], state[k], rtol=0, atol=0)


def test_mixed_accumulation_preserves_clean_core_gradients():
    model, controller, opt = setup()
    backward(model, controller, False)
    core_grads = {n: p.grad.clone() for n, p in model.named_parameters() if not p.gram_role}
    backward(model, controller, True, micro=1)
    for n, p in model.named_parameters():
        if not p.gram_role:
            torch.testing.assert_close(p.grad, core_grads[n], rtol=0, atol=0)
    assert controller.participation == [True, True]
    opt.step()
    assert controller.participation == [False, False]


def test_core_robustness_updates_aux_and_core():
    model, controller, _ = setup()
    backward(model, controller, False, robust=1)
    assert all(p.grad.abs().sum() > 0 for p in model.parameters())


def test_ablated_output_is_original_core():
    torch.manual_seed(9)
    model = Network()
    x = torch.tensor([[1, 2, 3]])
    expected = model(x).detach()
    controller = gram.install(model, args())
    controller.select(False, 0, 1, 0, 0, training=False)
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
    plain = Network()
    plain.load_state_dict(gram.core_state_dict(model.state_dict()), strict=True)
    torch.testing.assert_close(plain(x), model(x), rtol=0, atol=0)


def test_activation_checkpoint_recomputation_keeps_routing():
    model, controller, _ = setup()
    controller.select(True, 0, 1234, 0, 0)
    hidden = model.embed(torch.tensor([[1, 2, 3]])).detach().requires_grad_()
    checkpoint(model.layers, hidden, use_reentrant=True).square().mean().backward()
    assert model.layers[0].mlp.gram_aux[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None or p.grad.count_nonzero() == 0
               for p in model.parameters() if not p.gram_role)


def test_resume_matches_uninterrupted_steps():
    model, controller, opt = setup()
    backward(model, controller, True)
    opt.step()
    saved_model, saved_opt = copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict())
    opt.zero_grad(set_to_none=True)
    backward(model, controller, False, robust=1)
    opt.step()
    other, route2, opt2 = setup()
    other.load_state_dict(saved_model)
    opt2.load_state_dict(saved_opt)
    backward(other, route2, False, robust=1)
    opt2.step()
    for a, b in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_label_alignment_and_binary_validation(tmp_path):
    path = tmp_path / "labels.npy"
    dataset = [{"text": np.arange(4)} for _ in range(3)]
    np.save(path, np.array([0, 1, 0], dtype=bool))
    wrapped = gram.FlaggedDataset(dataset, path)
    assert wrapped[1]["gram_flag"] == 1
    np.save(path, np.array([0, 1]))
    with pytest.raises(ValueError, match="one entry"):
        gram.FlaggedDataset(dataset, path)
    np.save(path, np.array([0, 2, 0]))
    with pytest.raises(ValueError, match="only 0"):
        gram.FlaggedDataset(dataset, path)


def test_document_labels_follow_shuffle_and_packing_including_target(tmp_path):
    # Physical documents 0 and 2 are clean; 1 is flagged. doc_idx itself
    # reorders them, then shuffle_idx reorders the packed training samples.
    class Packed:
        indexed_dataset = SimpleNamespace(sizes=np.array([5, 5, 5]))
        doc_idx = np.array([2, 0, 1, 2])
        sample_idx = np.array([[0, 0], [1, 2], [2, 0], [3, 1]])
        shuffle_idx = np.array([2, 0, 1])

        def __len__(self):
            return 3

        def __getitem__(self, i):
            return {"text": np.array([i])}

    path = tmp_path / "docs.npy"
    np.save(path, np.array([0, 1, 0], dtype=np.uint8))
    wrapped = gram.FlaggedDataset(Packed(), path, "document")
    # Sample 1's final next-token target is at offset 0 of flagged doc 1.
    assert [int(wrapped[i]["gram_flag"]) for i in range(3)] == [1, 0, 1]


def test_document_labels_follow_precomputed_boundaries(tmp_path):
    class Packed:
        indexed_dataset = SimpleNamespace(sizes=np.array([5, 5, 5]))
        doc_idx = np.array([2, 0, 1, 2])
        sample_bounds = np.array([[0, 0, 1, 2], [1, 2, 2, 0], [2, 0, 3, 1]])

        def __len__(self):
            return 3

        def __getitem__(self, i):
            return {"text": np.array([i])}

    path = tmp_path / "docs.npy"
    np.save(path, np.array([0, 1, 0], dtype=np.uint8))
    wrapped = gram.FlaggedDataset(Packed(), path, "document")
    assert [int(wrapped[i]["gram_flag"]) for i in range(3)] == [0, 1, 1]


@pytest.mark.parametrize("kw", [{"precision": "fp16"}, {"zero_optimization": {"stage": 2}},
                                {"te_layernorm_mlp": True}, {"model_parallel_size": 2},
                                {"is_pipe_parallel": True}, {"gram_aux_width": 0}])
def test_unsupported_config_fails_before_training(kw):
    with pytest.raises(ValueError):
        gram.install(Network(), args(**kw))


def test_bf16_and_zero1_config_is_supported():
    gram.install(Network(), args(precision="bfloat16", zero_optimization={"stage": 1},
                                zero_allow_untested_optimizer=True))


def test_flat_master_groups_keep_roles_and_do_not_reset_between_zero_groups():
    model, controller, _ = setup()
    optimizer = gram.RoutedAdamW(model.parameters(), controller, managed_step=True,
                                 lr=0.01, weight_decay=0.1)
    groups = optimizer.param_groups
    # DeepSpeed replaces original Parameters with master partitions, retaining
    # group metadata. The replacement tensors deliberately have no gram_role.
    for group in groups:
        flat = nn.Parameter(torch.cat([p.detach().flatten() for p in group["params"]]))
        group["params"] = [flat]
    before = [g["params"][0].detach().clone() for g in groups]
    controller.participation[:] = [False, True]
    controller.prepare_step("cpu")
    for group in groups:
        group["params"][0].grad = torch.ones_like(group["params"][0])
        optimizer.param_groups = [group]  # ZeRO calls Adam once per group.
        optimizer.step()
    optimizer.param_groups = groups
    assert controller.participation == [False, True]
    for group, old in zip(groups, before):
        p = group["params"][0]
        state = optimizer.state[p]
        assert state["step"].item() == group["gram_role"]
        if not group["gram_role"]:
            torch.testing.assert_close(p, old, rtol=0, atol=0)
            assert state["exp_avg"].count_nonzero() == 0
        else:
            assert not torch.equal(p, old)
    controller.finish_step()
    assert controller.participation == [False, False]
    # Previously inactive state must remain usable when that role starts learning.
    controller.participation[:] = [True, False]
    controller.prepare_step("cpu")
    for group in groups:
        group["params"][0].grad = torch.ones_like(group["params"][0])
        optimizer.param_groups = [group]
        optimizer.step()
    optimizer.param_groups = groups
    assert all(optimizer.state[g["params"][0]]["step"].item() == 1 for g in groups)
    controller.finish_step()


def test_bf16_parameter_hooks_preserve_core_isolation():
    model, controller, _ = setup()
    model.to(torch.bfloat16)
    backward(model, controller, True)
    assert all(p.grad is not None and p.grad.count_nonzero() == 0
               for p in model.parameters() if not p.gram_role)
    assert model.layers[0].mlp.gram_aux[0].weight.grad.abs().sum() > 0
