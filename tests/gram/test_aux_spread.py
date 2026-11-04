"""Paper p_as (auxiliary spread): flagged microbatches update the core with probability p_as."""
import pytest
import torch

from test_routing import Network, args, gram


def setup(**kw):
    torch.manual_seed(7)
    model = Network()
    controller = gram.install(model, args(**kw))
    return model, controller


def roles(controller, flagged, spread, micro, robustness=0.0, training=True):
    controller.select(flagged, robustness, 1234, 5, micro, training=training, spread=spread)
    return tuple(controller.train_roles), controller.aux_on


def test_spread_zero_is_strict_isolation_and_default():
    _, c = setup()
    for micro in range(200):
        assert roles(c, 1, 0.0, micro) == ((False, True), True)
        c.select(1, 0.0, 1234, 5, micro)  # positional call without spread keeps old behaviour
        assert tuple(c.train_roles) == (False, True)


def test_spread_one_trains_core_and_aux_on_flagged():
    model, c = setup()
    assert roles(c, 1, 1.0, 0) == ((True, True), True)
    model(torch.tensor([[1, 2, 3, 4]])).square().mean().backward()
    assert all(p.grad.abs().sum() > 0 for p in model.parameters())


def test_spread_rate_is_close_to_p_and_independent_of_robustness_draw():
    _, c = setup()
    core = [roles(c, 1, 0.5, m)[0][0] for m in range(4000)]
    assert abs(sum(core) / len(core) - 0.5) < 0.03
    # clean data: robustness draw must not be correlated with the spread draw
    pairs = []
    for m in range(4000):
        c.select(1, 0.5, 1234, 5, m, spread=0.5); s = c.train_roles[0]
        c.select(0, 0.5, 1234, 5, m, spread=0.5); r = c.aux_on
        pairs.append((s, r))
    both = sum(s and r for s, r in pairs) / len(pairs)
    assert abs(both - 0.25) < 0.03


def test_spread_never_affects_clean_data_or_evaluation():
    _, c = setup()
    for m in range(200):
        (core, aux), _ = roles(c, 0, 1.0, m, robustness=0.0)
        assert core and not aux
        assert roles(c, 0, 1.0, m, training=False) == roles(c, 0, 0.0, m, training=False)
        assert roles(c, 1, 1.0, m)[0] != roles(c, 1, 0.0, m)[0]


def test_nested_spread_updates_less_restrictive_prefixes_only():
    _, c = setup(gram_nested=True, gram_inner_labels_path="weak.npy", gram_optimizer_layout=3)
    assert roles(c, 1, 1.0, 0)[0] == (True, True, False)
    assert roles(c, 2, 1.0, 0)[0] == (True, True, True)
    assert roles(c, 1, 0.0, 0)[0] == (False, True, False)
    assert roles(c, 2, 0.0, 0)[0] == (False, False, True)


def test_resume_stable_spread_draws():
    _, c = setup()
    first = [roles(c, 1, 0.5, m)[0] for m in range(64)]
    _, d = setup()
    assert first == [roles(d, 1, 0.5, m)[0] for m in range(64)]


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_invalid_spread_rejected(bad):
    with pytest.raises(ValueError):
        gram.validate_config(args(gram_aux_spread=bad))


def test_grouped_batches_respect_spread():
    a = args(gram_aux_spread=1.0, gram_nested=False, seed=1234, iteration=3, gram_grouped_microbatch_size=4)
    batches = [dict(text=torch.zeros(1, 5, dtype=torch.long), gram_flag=torch.tensor([f])) for f in (1, 1, 0, 0)]
    groups = gram.grouped_batches(batches, a)
    # with spread=1 and robustness drawn per micro, flagged examples all share roles (core+aux)
    flagged_groups = [g for g in groups if int(g[2]["gram_flag"][0]) == 1]
    assert len(flagged_groups) == 1 and flagged_groups[0][1] == 2
