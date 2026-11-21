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

"""μP (https://arxiv.org/abs/2203.03466) and CompleteP (https://arxiv.org/abs/2505.01618).

Hidden weights are the matrices inside transformer layers. Their init std is divided by
sqrt(width_mult) and their Adam/Lion learning rate by width_mult. Embeddings, the readout,
and the final norm keep the base init and learning rate, and the readout input is divided
by width_mult. CompleteP also multiplies each residual branch by depth_mult ** -alpha and
the learning rate of every hidden parameter by depth_mult ** (alpha - 1).
"""

from collections import defaultdict

from megatron.model.init_functions import get_init_methods, scaled_init_method
from megatron.model.utils import get_params_for_weight_decay_optimization


def width_mult(neox_args):
    if neox_args.parameterization == "sp":
        return 1.0
    return neox_args.hidden_size / neox_args.mup_base_hidden_size


def depth_mult(neox_args):
    if neox_args.parameterization != "completep":
        return 1.0
    return neox_args.num_layers / neox_args.mup_base_num_layers


def depth_alpha(neox_args):
    return (
        neox_args.completep_alpha if neox_args.parameterization == "completep" else 1.0
    )


def residual_multiplier(neox_args):
    return depth_mult(neox_args) ** -depth_alpha(neox_args)


def output_multiplier(neox_args):
    if neox_args.parameterization == "sp":
        return 1.0
    return 1.0 / (neox_args.mup_output_temp * width_mult(neox_args))


def get_init_methods_for_parameterization(neox_args):
    """Returns init methods for (embeddings and readout, hidden inputs, hidden outputs)."""
    if neox_args.parameterization == "sp":
        init_method, output_init_method = get_init_methods(neox_args)
        return init_method, init_method, output_init_method
    num_layers = (
        neox_args.mup_base_num_layers
        if neox_args.parameterization == "completep"
        else neox_args.num_layers
    )
    init_method, output_init_method = get_init_methods(
        neox_args, neox_args.mup_base_hidden_size, num_layers
    )
    scale = width_mult(neox_args) ** -0.5
    return (
        init_method,
        scaled_init_method(init_method, scale),
        scaled_init_method(output_init_method, scale),
    )


def get_param_groups(model, neox_args):
    """Optimizer param groups. Under mup and completep each group carries an lr_scale,
    which the LR scheduler applies, and its own weight_decay and eps."""
    groups = get_params_for_weight_decay_optimization(model, neox_args)
    if neox_args.parameterization == "sp":
        return groups

    from megatron.model.transformer import ParallelTransformerLayer

    hidden = {
        id(p)
        for m in model.modules()
        if isinstance(m, ParallelTransformerLayer)
        for p in m.parameters()
    }
    m, d, alpha = width_mult(neox_args), depth_mult(neox_args), depth_alpha(neox_args)
    eps = neox_args.optimizer["params"].get("eps", 1e-8)
    wd_mult = m if neox_args.mup_independent_weight_decay else 1.0

    def hyperparameters(p, weight_decay):
        if id(p) not in hidden:
            return 1.0, weight_decay, eps / m
        if p.ndim < 2:
            return d ** (alpha - 1), weight_decay, eps / m * d**-alpha
        return d ** (alpha - 1) / m, weight_decay * wd_mult, eps / m * d**-alpha

    params = defaultdict(list)
    for group in groups:
        weight_decay = group.get("weight_decay", neox_args.weight_decay)
        for p in group["params"]:
            params[hyperparameters(p, weight_decay)].append(p)

    with_eps = neox_args.optimizer_type.lower() == "adam"
    return [
        dict(
            params=ps,
            lr_scale=lr_scale,
            weight_decay=weight_decay,
            **({"eps": group_eps} if with_eps else {}),
        )
        for (lr_scale, weight_decay, group_eps), ps in params.items()
    ]
