# Copyright (c) 2026, EleutherAI
# SPDX-License-Identifier: Apache-2.0

"""Value ranges shared by config loading and direct generation requests."""

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


def validate_sampling_options(options, *, already_validated=None):
    """Check only values not covered by the validated config snapshot."""
    for name, value in options.items():
        if already_validated is not None:
            previous = already_validated[name]
            if type(value) is type(previous) and value == previous:
                continue
        if name == "maximum_tokens" and value is None:
            continue  # Infer the budget from the prompt and context window.
        accepts, domain = SAMPLING_RANGES[name]
        if isinstance(value, bool) or not isinstance(value, Real) or not accepts(value):
            raise ValueError(f"{name} must be {domain}; got {value!r}")
