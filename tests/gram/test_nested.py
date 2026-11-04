"""Three disjoint labels, strict backward roles, and removable nested prefixes."""
import copy

import numpy as np
import pytest
import torch
from torch.utils.checkpoint import checkpoint

from test_routing import Network, args, gram


def setup():
    torch.manual_seed(7)
    model = Network()
    controller = gram.install(model, args(gram_nested=True, gram_inner_labels_path="weak.npy",
                                          gram_optimizer_layout=3, gram_aux_width=1024))
    optimizer = gram.RoutedAdamW(model.parameters(), controller, lr=0.01, weight_decay=0.1)
    return model, controller, optimizer


def backward(model, route, label, iteration=0, micro=0, robustness=0):
    route.select(label, robustness, 1234, iteration, micro)
    model(torch.tensor([[1, 2, 3, 4]])).square().mean().backward()


@pytest.mark.parametrize("label", [0, 1, 2])
def test_only_intended_role_changes_even_after_other_roles_have_momentum(label):
    model, route, opt = setup()
    # Seed nonzero momentum in every role before testing a frozen role.
    for warm_label in (0, 1, 2):
        backward(model, route, warm_label)
        opt.step()
        opt.zero_grad(set_to_none=True)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    states = {p: copy.deepcopy(s) for p, s in opt.state.items()}
    backward(model, route, label)
    if label == 2:
        assert route.aux_on and route.inner_on  # Outer participates but must not learn.
        assert model.layers[0].mlp.gram_inner[0].weight.grad.abs().sum() > 0
    opt.step()
    changed = [False] * 3
    for name, param in model.named_parameters():
        if param.gram_role != label:
            torch.testing.assert_close(param, before[name], rtol=0, atol=0)
            for key, old in states[param].items():
                torch.testing.assert_close(opt.state[param][key], old, rtol=0, atol=0)
        else:
            changed[label] |= not torch.equal(param, before[name])
    assert changed[label]


def test_inner_batch_preserves_previously_accumulated_outer_and_core_gradients():
    model, route, opt = setup()
    backward(model, route, 0, micro=0)
    backward(model, route, 1, micro=1)
    before = {n: p.grad.clone() for n, p in model.named_parameters() if p.gram_role != 2}
    backward(model, route, 2, micro=2)
    for name, param in model.named_parameters():
        if param.gram_role != 2:
            torch.testing.assert_close(param.grad, before[name], rtol=0, atol=0)
    assert route.participation == [True, True, True]
    opt.step()
    assert route.participation == [False, False, False]


def test_all_three_prefixes_match_physically_reduced_models():
    model, route, _ = setup()
    x = torch.tensor([[1, 2, 3, 4]])
    state = model.state_dict()
    for level in (0, 1, 2):
        route.select(0, 0, 1234, 0, 0, training=False, level=level)
        expected = model(x)
        other = Network()
        if level == 1:
            controller = gram.install(other, args(gram_aux_width=1024))
        elif level == 2:
            controller = gram.install(other, args(gram_aux_width=1024, gram_nested=True,
                                                 gram_inner_labels_path="weak.npy", gram_optimizer_layout=3))
        other.load_state_dict(gram.prefix_state_dict(state, level), strict=True)
        if level:
            controller.select(0, 0, 1234, 0, 0, training=False, level=level)
        torch.testing.assert_close(other(x), expected, rtol=0, atol=0)
        assert route.participation == [False, False, False]


def test_inner_gradient_survives_activation_checkpoint_and_frozen_outer():
    model, route, _ = setup()
    route.select(2, 0, 1234, 0, 0)
    hidden = model.embed(torch.tensor([[1, 2, 3]])).detach().requires_grad_()
    checkpoint(model.layers, hidden, use_reentrant=True).square().mean().backward()
    assert hidden.grad.abs().sum() > 0
    assert model.layers[0].mlp.gram_inner[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None or p.grad.count_nonzero() == 0
               for p in model.parameters() if p.gram_role != 2)


def test_nested_label_join_rejects_non_nested_filters_and_packed_inner_takes_precedence(tmp_path):
    class Packed:
        indexed_dataset = type("Store", (), {"sizes": np.ones(3, dtype=int)})()
        doc_idx = np.array([2, 0, 1])
        sample_bounds = np.array([[0, 0, 0, 0], [1, 0, 2, 0], [2, 0, 2, 0]])

        def __len__(self):
            return 3

        def __getitem__(self, index):
            return {"text": np.array([index])}

    strong, weak = tmp_path / "strong.npy", tmp_path / "weak.npy"
    np.save(strong, np.array([1, 1, 0], dtype=np.uint8))
    np.save(weak, np.array([0, 1, 0], dtype=np.uint8))
    ds = gram.FlaggedDataset(Packed(), strong, "document", weak)
    assert [ds[i]["gram_flag"] for i in range(3)] == [0, 2, 2]
    ds = gram.FlaggedDataset([{}] * 3, strong, "sample", weak)
    assert [ds[i]["gram_flag"] for i in range(3)] == [1, 2, 0]
    np.save(weak, np.array([0, 1, 1], dtype=np.uint8))
    with pytest.raises(ValueError, match="subset"):
        gram.FlaggedDataset([{}] * 3, strong, "sample", weak)


def test_nested_checkpoint_resume_and_wrong_layout_rejection():
    model, route, opt = setup()
    backward(model, route, 2)
    opt.step()
    saved_model, saved_opt = copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict())
    opt.zero_grad(set_to_none=True)
    for label in (1, 0, 2):
        backward(model, route, label)
        opt.step()
        opt.zero_grad(set_to_none=True)
    other, other_route, other_opt = setup()
    other.load_state_dict(saved_model)
    other_opt.load_state_dict(saved_opt)
    for label in (1, 0, 2):
        backward(other, other_route, label)
        other_opt.step()
        other_opt.zero_grad(set_to_none=True)
    for a, b in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    plain = Network()
    one = gram.install(plain, args())
    one_opt = gram.RoutedAdamW(plain.parameters(), one, lr=0.01)
    with pytest.raises(ValueError, match="role layout"):
        one_opt.load_state_dict(saved_opt)


def test_robustness_only_uses_valid_prefixes_and_resume_stable_draws():
    _, route, _ = setup()
    observed = set()
    for step in range(32):
        route.select(0, 1, 1234, step, 0)
        active = tuple(route.train_roles)
        assert active in ((True, True, False), (True, True, True))
        route.select(0, 1, 1234, step, 0)
        assert tuple(route.train_roles) == active
        observed.add(active)
    assert len(observed) == 2
    for label in (1, 2):
        route.select(label, 1, 1234, 0, 0)
        assert route.train_roles == [i == label for i in range(3)]


def test_default_inference_profile_and_invalid_route_rejection():
    model = Network()
    route = gram.install(model, args(gram_nested=True, gram_inner_labels_path="weak.npy",
                                     gram_optimizer_layout=3, gram_eval_level=1))
    assert route.aux_on and not route.inner_on
    assert route.participation == [False, False, False]
    with pytest.raises(ValueError):
        route.select(1.5, 0, 1234, 0, 0)
