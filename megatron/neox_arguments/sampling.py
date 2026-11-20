# Copyright (c) 2026, EleutherAI
# SPDX-License-Identifier: Apache-2.0

"""Value ranges shared by config loading and direct generation requests."""

import warnings
from math import isfinite
from numbers import Integral, Real


_COUNT = (lambda x: isinstance(x, Integral) and x >= 0, "a non-negative integer")
_PROBABILITY = (lambda x: 0 <= x <= 1, "in [0, 1]")
_CUTOFF = (lambda x: 0 <= x < 1, "in [0, 1)")
SAMPLING_RANGES = {
    "minimum_tokens": _COUNT,
    "maximum_tokens": _COUNT,
    "temperature": (lambda x: isfinite(x) and x >= 0, "finite and non-negative"),
    "top_k": _COUNT,
    "top_p": _PROBABILITY,
    "min_p": _PROBABILITY,
    "top_h": _PROBABILITY,
    "typical_p": (lambda x: 0 < x <= 1, "in (0, 1]"),
    "epsilon_cutoff": _CUTOFF,
    "eta_cutoff": _CUTOFF,
}

# Active warpers only: neither token budgets nor the EOS mask is ignored by greedy.
_NEUTRAL_FILTERS = {
    "top_h": (0,),
    "top_k": (0,),
    "top_p": (0, 1),
    "min_p": (0,),
    "typical_p": (1,),
    "epsilon_cutoff": (0,),
    "eta_cutoff": (0,),
}


def validate_sampling_options(options):
    """Range-check previously unseen values."""
    for name, value in options.items():
        if name == "maximum_tokens" and value is None:
            continue  # Infer the budget from the prompt and context window.
        accepts, domain = SAMPLING_RANGES[name]
        if isinstance(value, bool) or not isinstance(value, Real) or not accepts(value):
            raise ValueError(f"{name} must be {domain}; got {value!r}")


def prepare_sampling_options(options, *, already_validated=None):
    """Check a config or request at entry; leave unchanged config values alone."""
    changed = (
        options
        if already_validated is None
        else {
            name: value
            for name, value in options.items()
            if type(value) is not type(already_validated[name])
            or value != already_validated[name]
        }
    )
    if not changed:
        return
    validate_sampling_options(changed)
    if "minimum_tokens" in changed or "maximum_tokens" in changed:
        minimum, maximum = options["minimum_tokens"], options["maximum_tokens"]
        if maximum is not None and minimum > maximum:
            raise ValueError(
                f"minimum_tokens ({minimum}) cannot exceed maximum_tokens ({maximum})"
            )
    if options["temperature"] == 0 and (
        "temperature" in changed or _NEUTRAL_FILTERS.keys() & changed.keys()
    ):
        ignored = [
            f"{name}={options[name]}"
            for name, neutral in _NEUTRAL_FILTERS.items()
            if options[name] not in neutral
        ]
        if ignored:
            warnings.warn(
                "temperature=0 selects greedy decoding; these sampling filters will be ignored: "
                f"{', '.join(ignored)}. Use a positive temperature to enable sampling.",
                UserWarning,
                stacklevel=2,
            )


def cap_top_k(top_k, vocab_size):
    """The vocabulary bound is only known once the model or logits are available."""
    if top_k > vocab_size:
        warnings.warn(
            f"top_k={top_k} exceeds the vocabulary size ({vocab_size}); "
            f"truncating top_k to {vocab_size}.",
            UserWarning,
            stacklevel=2,
        )
        return vocab_size
    return top_k
