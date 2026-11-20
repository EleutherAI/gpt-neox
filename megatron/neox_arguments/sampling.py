# Copyright (c) 2026, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

# Only these worker overrides can change which sampling filters greedy ignores.
GREEDY_WARNING_FIELDS = {"temperature", *_NEUTRAL_FILTERS}


class IgnoredSamplingFiltersWarning(UserWarning):
    """Greedy decoding was requested with active sampling filters."""


def validate_sampling_options(options):
    """Range-check the supplied sampling fields."""
    for name, value in options.items():
        if name == "maximum_tokens" and value is None:
            continue  # Infer the budget from the prompt and context window.
        accepts, domain = SAMPLING_RANGES[name]
        if isinstance(value, bool) or not isinstance(value, Real) or not accepts(value):
            raise ValueError(f"{name} must be {domain}; got {value!r}")


def validate_generation_options(options):
    """Validate a complete set of options at config load or public request entry."""
    validate_sampling_options(options)
    minimum, maximum = options["minimum_tokens"], options["maximum_tokens"]
    if maximum is not None and minimum > maximum:
        raise ValueError(
            f"minimum_tokens ({minimum}) cannot exceed maximum_tokens ({maximum})"
        )
    if options["temperature"] == 0:
        ignored = [
            f"{name}={options[name]}"
            for name, neutral in _NEUTRAL_FILTERS.items()
            if options[name] not in neutral
        ]
        if ignored:
            warnings.warn(
                "temperature=0 selects greedy decoding; these sampling filters will be ignored: "
                f"{', '.join(ignored)}. Use a positive temperature to enable sampling.",
                IgnoredSamplingFiltersWarning,
                stacklevel=2,
            )


def sampling_options_from_config(neox_args):
    return {name: getattr(neox_args, name) for name in SAMPLING_RANGES}


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
