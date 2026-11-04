"""Opt-in GRAM routing for NeoX data-parallel FP32/BF16 training.

Gradient hooks mask *parameter* gradients, preserving activation derivatives.
Optimizer roles survive DeepSpeed flattening and ZeRO stage-1 partitioning.
"""
import hashlib

import numpy as np
import torch
from torch import nn


class FlaggedDataset(torch.utils.data.Dataset):
    """Labels index the final training dataset, including its shuffle/blend order."""

    def __init__(self, dataset, labels_path, granularity="sample", inner_labels_path=None):
        self.dataset = dataset
        self.granularity = granularity
        self.labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
        if granularity == "sample":
            expected = len(dataset)
        elif granularity == "document":
            if not hasattr(dataset, "indexed_dataset") or not hasattr(dataset, "doc_idx"):
                raise ValueError("Document routing requires one GPT2 or precomputed dataset, not a blend")
            expected = dataset.indexed_dataset.sizes.shape[0]
        else:
            raise ValueError("GRAM label granularity must be sample or document")
        if self.labels.shape != (expected,):
            raise ValueError(f"GRAM labels must have one entry per final training {granularity}")
        if not np.isin(self.labels, [0, 1]).all():
            raise ValueError("GRAM labels must contain only 0 (core) and 1 (flagged)")
        self.inner_labels = None
        if inner_labels_path:
            self.inner_labels = np.load(inner_labels_path, mmap_mode="r", allow_pickle=False)
            if self.inner_labels.shape != (expected,) or not np.isin(self.inner_labels, [0, 1]).all():
                raise ValueError("Inner labels must be binary and match the outer label shape")
            if np.any(self.inner_labels > self.labels):
                raise ValueError("Inner/weak filter must be a subset of outer/strong filter")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.granularity == "sample":
            flagged = self.labels[index]
            inner = self.inner_labels[index] if self.inner_labels is not None else False
        else:
            if hasattr(self.dataset, "sample_bounds"):
                first, _, last, _ = self.dataset.sample_bounds[index]
            else:
                shuffled = self.dataset.shuffle_idx[index]
                first = self.dataset.sample_idx[shuffled, 0]
                last = self.dataset.sample_idx[shuffled + 1, 0]
            # A packed sequence containing any flagged document is routed in
            # its entirety, including its next-token target at the boundary.
            docs = self.dataset.doc_idx[int(first):int(last) + 1]
            flagged = self.labels[docs].any()
            inner = self.inner_labels[docs].any() if self.inner_labels is not None else False
        # 0=clean, 1=strong-only/outer, 2=weak/inner. A packed inner target
        # takes precedence over outer or clean text anywhere in the sample.
        return dict(self.dataset[index], gram_flag=np.int64(2 if inner else flagged))


class RoutingController:
    def __init__(self, nested=False, inplace_mask=False):
        self.nested = nested
        self.inplace_mask = inplace_mask
        self.num_roles = 3 if nested else 2
        self.core_on = True
        self.aux_on = False
        self.inner_on = False
        self.train_roles = [True] + [False] * (self.num_roles - 1)
        self.participation = [False] * self.num_roles
        self.step_active = None
        self.handles = []

    def prepare_step(self, device, group=None):
        """One collective per global update, before DeepSpeed visits groups."""
        active = torch.tensor(self.participation, device=device, dtype=torch.int32)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(active, op=torch.distributed.ReduceOp.MAX, group=group)
        self.step_active = active.tolist()

    def finish_step(self):
        self.participation[:] = [False] * self.num_roles
        self.step_active = None

    def select(self, flagged, robustness, seed, iteration, microstep, training=True, level=0, spread=0.0):
        """Choose forward branches and parameter-update roles for one microbatch.

        robustness (paper p_cr): on clean data, probability of enabling and
        training an auxiliary prefix. spread (paper p_as): on flagged data,
        probability that the less restrictive parameters (core, and the outer
        bank for inner data) are also updated. spread=0 is strict isolation.
        """
        # Stateless draws: resume does not require serializing a second RNG stream.
        key = f"{seed}:{iteration}:{microstep}".encode()
        digest = hashlib.sha256(key).digest()
        draw = int.from_bytes(digest[:8], "big") / 2**64
        # Independent of the robustness draw and of the nested prefix choice (bytes 8:16).
        spread_on = training and int.from_bytes(digest[16:24], "big") / 2**64 < spread
        if self.nested:
            if flagged not in (0, 1, 2):
                raise ValueError("Nested GRAM route must be 0, 1, or 2")
            route = int(flagged)
            if route not in (0, 1, 2) or level not in (0, 1, 2):
                raise ValueError("Nested GRAM route and evaluation level must be 0, 1, or 2")
            forward_level = route if training else level
            if training and route == 0 and draw < robustness:
                # Robustness samples only valid nested prefixes. Clean data
                # may train enabled adapters; restricted data never trains
                # a less-filtered prefix's parameters.
                forward_level = 1 + (int.from_bytes(hashlib.sha256(key).digest()[8:16], "big") % 2)
            self.aux_on = forward_level >= 1
            self.inner_on = forward_level >= 2
            self.train_roles = ([route == 0 or (route >= 1 and spread_on),
                                 route == 1 or (route == 0 and self.aux_on) or (route == 2 and spread_on),
                                 route == 2 or (route == 0 and self.inner_on)]
                                if training else [False] * 3)
            self.core_on = self.train_roles[0]
        else:
            if flagged not in (0, 1) or level not in (0, 1):
                raise ValueError("Single-bank GRAM route and evaluation level must be 0 or 1")
            self.core_on = not flagged or spread_on
            self.aux_on = bool(flagged or (training and draw < robustness) or (not training and level == 1))
            self.train_roles = [self.core_on, self.aux_on]
        if training:
            for role in range(self.num_roles):
                self.participation[role] |= self.train_roles[role]

    def attach(self, model):
        for name, param in model.named_parameters():
            role = 2 if ".gram_inner." in name else (1 if ".gram_aux." in name else 0)
            param.gram_role = role
            if not hasattr(param, "model_parallel"):
                param.model_parallel = False
            def mask(grad, role=role):
                enabled = self.train_roles[role]
                if enabled:
                    return grad
                return grad.zero_() if self.inplace_mask else torch.zeros_like(grad)
            self.handles.append(param.register_hook(mask))


class AuxiliaryMLP(nn.Module):
    def __init__(self, core, hidden_size, width, controller, init_std, output_bias=True):
        super().__init__()
        self.core = core
        self.controller = controller
        ref = next(core.parameters())
        self.gram_aux = nn.Sequential(
            nn.Linear(hidden_size, width), nn.GELU(), nn.Linear(width, hidden_size, bias=output_bias)
        ).to(device=ref.device, dtype=ref.dtype)
        self.gram_inner = (nn.Sequential(
            nn.Linear(hidden_size, width), nn.GELU(), nn.Linear(width, hidden_size, bias=output_bias)
        ).to(device=ref.device, dtype=ref.dtype) if controller.nested else None)
        branches = [self.gram_aux] + ([self.gram_inner] if self.gram_inner is not None else [])
        for module in (m for branch in branches for m in branch):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_states):
        output, bias = self.core(hidden_states)
        # Keep the graph identical for all ranks, including inactive auxiliary
        # branches. This prototype prioritizes correctness over kernel efficiency.
        aux = self.gram_aux(hidden_states)
        output = output + aux * float(self.controller.aux_on)
        if self.gram_inner is not None:
            output = output + self.gram_inner(hidden_states) * float(self.controller.inner_on)
        return output, bias


def role_parameter_groups(params):
    """Never flatten core and auxiliary weights into the same master tensor."""
    params = list(params)
    if not params:
        raise ValueError("GRAM optimizer received no parameters")
    groups = params if isinstance(params[0], dict) else [{"params": params}]
    result = []
    for group in groups:
        members = list(group["params"])
        if any(not hasattr(p, "gram_role") for p in members):
            raise ValueError("Install GRAM before constructing its optimizer")
        roles = sorted({p.gram_role for p in members})
        if any(role not in (0, 1, 2) for role in roles):
            raise ValueError("Unknown GRAM optimizer role")
        for role in roles:
            selected = [p for p in members if p.requires_grad and p.gram_role == role]
            if selected:
                split = dict(group, params=selected, gram_role=role)
                if "name" in split:
                    split["name"] = ("core_", "aux_", "inner_")[role] + split["name"]
                result.append(split)
    return result


class RoutedAdamW(torch.optim.AdamW):
    """Skip all state, momentum and decay updates for globally inactive roles."""

    def __init__(self, params, controller, managed_step=False, **kwargs):
        super().__init__(role_parameter_groups(params), **kwargs)
        self.controller = controller
        self.managed_step = managed_step
        self._routing_groups = self.param_groups

    def load_state_dict(self, state_dict):
        saved_roles = [g.get("gram_role") for g in state_dict["param_groups"]]
        if saved_roles != [g.get("gram_role") for g in self.param_groups]:
            raise ValueError("GRAM optimizer checkpoint role layout does not match model")
        super().load_state_dict(state_dict)
        self._routing_groups = self.param_groups

    def _initialize_master_states(self):
        # DeepSpeed links every partition's moment buffers after the first step,
        # including roles that have not yet participated. Allocate zero moments
        # with step=0, without applying a fake update, decay, or momentum step.
        for group in self._routing_groups:
            for p in group["params"]:
                state = self.state[p]
                if not state:
                    step_device = p.device if group.get("capturable") or group.get("fused") else "cpu"
                    state["step"] = torch.zeros((), dtype=torch.float32, device=step_device)
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group.get("amsgrad"):
                        state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("GRAM optimizer does not support closures")
        if self.managed_step:
            if self.controller.step_active is None:
                raise RuntimeError("Prepare GRAM activity before DeepSpeed's optimizer step")
            self._initialize_master_states()
        else:
            param = next(p for group in self.param_groups for p in group["params"])
            self.controller.prepare_step(param.device)
        active = self.controller.step_active
        for group in self.param_groups:
            for p in group["params"]:
                # BF16 and ZeRO replace p with a new flat/partitioned FP32
                # tensor. Group metadata, unlike Parameter attributes, survives.
                if not active[group["gram_role"]]:
                    p.grad = None
        result = super().step()
        if not self.managed_step:
            self.controller.finish_step()
        return result


def absorbed_mlp_width(original, auxiliary, banks, alignment):
    """Reallocate MLP channels, preserving weights + biases exactly.

    Absorbed adapters omit output biases: the single core output bias remains.
    """
    width = original - banks * auxiliary
    if width <= 0 or original % alignment or width % alignment:
        raise ValueError("Absorbed MLP widths must be positive and preserve NeoX alignment")
    return width


def validate_config(args):
    unsupported = []
    for name, enabled in {
        "pipeline parallelism": args.is_pipe_parallel,
        "tensor parallelism": args.model_parallel_size != 1,
        "unsupported precision": args.precision not in ("fp32", "bfloat16"),
        "ZeRO stage other than 0/1": (args.zero_optimization or {}).get("stage", 0) not in (0, 1),
        "optimizer offload": bool((args.zero_optimization or {}).get("cpu_offload"))
            or bool((args.zero_optimization or {}).get("offload_optimizer")),
        "Transformer Engine fused MLP": args.te_layernorm_mlp,
        "Transformer Engine attention": args.te_mha,
        "MoE": args.moe_num_experts > 1,
        "non-normal training": args.train_impl != "normal",
        "microbatch > 1": args.train_micro_batch_size_per_gpu != 1,
        "missing sample labels": not args.gram_labels_path,
        "muP": args.use_mup,
        "8-bit optimizer": args.use_bnb_optimizer,
        "non-Adam optimizer": args.optimizer_type.lower() != "adam",
        "frozen/soft-prompt tuning": bool(getattr(args, "soft_prompt_tuning", None)),
        "missing explicit optimizer params": args.optimizer is None,
        "online dataset": getattr(args, "dataset_impl", "gpt2") == "online",
        "optimizer reset/dummy mode": bool(getattr(args, "no_load_optim", False)),
        "checkpoint forward validation": bool(getattr(args, "checkpoint_validation_with_forward_pass", False)),
    }.items():
        if enabled:
            unsupported.append(name)
    if unsupported:
        raise ValueError("GRAM prototype does not support: " + ", ".join(unsupported))
    if args.gram_aux_width < 1 or not 0 <= args.gram_core_robustness <= 1:
        raise ValueError("GRAM requires positive width and robustness in [0, 1]")
    if not 0 <= getattr(args, "gram_aux_spread", 0.0) <= 1:
        raise ValueError("GRAM auxiliary spread must be in [0, 1]")
    grouped = getattr(args, "gram_grouped_microbatch_size", 1)
    if grouped < 1:
        raise ValueError("GRAM grouped microbatch size must be positive")
    if grouped > 1:
        if any(getattr(args, key, 0) for key in ("hidden_dropout", "attention_dropout")):
            raise ValueError("Grouped routing requires zero dropout to preserve per-example RNG semantics")
        if getattr(args, "curriculum_learning", None):
            raise ValueError("Grouped routing does not support curriculum learning")
    if getattr(args, "gram_absorb_auxiliary_width", False) and getattr(args, "activation", "gelu") != "gelu":
        raise ValueError("Absorbed width currently supports ordinary GELU MLPs")
    nested = getattr(args, "gram_nested", False)
    expected_layout = 3 if nested else 2
    if getattr(args, "gram_optimizer_layout", 2) != expected_layout:
        raise ValueError(f"GRAM requires optimizer layout version {expected_layout}")
    if nested and not getattr(args, "gram_inner_labels_path", None):
        raise ValueError("Nested GRAM requires inner/weak labels and outer/strong labels")
    if not nested and getattr(args, "gram_inner_labels_path", None):
        raise ValueError("Inner labels require gram_nested=true")
    if getattr(args, "gram_eval_level", 0) not in range(3 if nested else 2):
        raise ValueError("Invalid GRAM evaluation level")
    if (args.zero_optimization or {}).get("stage", 0) == 1 and not getattr(args, "zero_allow_untested_optimizer", False):
        raise ValueError("GRAM ZeRO-1 requires zero_allow_untested_optimizer: true for RoutedAdamW")


def install(model, args):
    validate_config(args)
    controller = RoutingController(nested=getattr(args, "gram_nested", False),
                                   inplace_mask=getattr(args, "gram_inplace_mask", False))
    replacements = []
    for module in list(model.modules()):
        if hasattr(module, "mlp") and type(module.mlp).__name__ == "ParallelMLP":
            replacements.append(module)
    if len(replacements) != args.num_layers:
        raise ValueError("GRAM expected one ordinary ParallelMLP per transformer layer")
    for module in replacements:
        module.mlp = AuxiliaryMLP(module.mlp, args.hidden_size, args.gram_aux_width,
                                  controller, args.init_method_std,
                                  output_bias=(getattr(args, "gram_aux_output_bias", True)
                                               and not getattr(args, "gram_absorb_auxiliary_width", False)))
    controller.attach(model)
    controller.select(0, 0, getattr(args, "seed", 0), 0, 0, training=False,
                      level=getattr(args, "gram_eval_level", 0))
    return controller


def core_state_dict(state_dict):
    """Convert a GRAM model state dict to ordinary NeoX names (no optimizer)."""
    result = {}
    for name, value in state_dict.items():
        if ".gram_aux." in name or ".gram_inner." in name:
            continue
        name = name.replace(".mlp.core.", ".mlp.")
        if name in result:
            raise ValueError(f"Duplicate core state key: {name}")
        result[name] = value
    return result


def prefix_state_dict(state_dict, level):
    """Weights for a valid nested prefix; levels 1/2 keep the GRAM wrapper.

    Level 0 loads in ordinary NeoX. Level 1 loads with gram_nested=false,
    gram_enabled=true. Level 2 loads with gram_nested=true. Training state
    must never accompany a reduced-prefix export.
    """
    if level not in (0, 1, 2):
        raise ValueError("GRAM prefix level must be 0, 1, or 2")
    if level == 0:
        return core_state_dict(state_dict)
    if level == 2 and not any(".gram_inner." in k for k in state_dict):
        raise ValueError("Level 2 requires a nested GRAM checkpoint")
    return {k: v for k, v in state_dict.items() if level == 2 or ".gram_inner." not in k}


def grouped_batches(batches, args):
    """Group only within one existing update, preserving each sample's route draw.

    Loader/sampler microbatch remains one. The representative original index
    reproduces the group's complete routing state when forward_step selects it.
    """
    probe = RoutingController(nested=args.gram_nested)
    buckets = {}
    for micro, batch in enumerate(batches):
        if batch['text'].shape[0] != 1 or batch['gram_flag'].numel() != 1:
            raise ValueError('Grouped routing expects the original singleton loader batches')
        label = int(batch['gram_flag'].item())
        probe.select(label, args.gram_core_robustness, args.seed, args.iteration, micro,
                     spread=getattr(args, 'gram_aux_spread', 0.0))
        key = (label, probe.aux_on, probe.inner_on, tuple(probe.train_roles))
        buckets.setdefault(key, []).append((micro, batch))
    maximum = args.gram_grouped_microbatch_size
    groups = []
    for members in buckets.values():
        for offset in range(0, len(members), maximum):
            part = members[offset:offset + maximum]
            keys = part[0][1].keys()
            if any(row.keys() != keys for _, row in part):
                raise ValueError('Grouped examples must have the same dataset columns')
            data = {key: torch.cat([row[key] for _, row in part], dim=0) for key in keys}
            groups.append((part[0][0], len(part), data))
    return groups


def grouped_train_step(args, timers, data_iterator, model, optimizer, lr_scheduler,
                       reference_model, forward_step, backward_step, reduce_losses):
    """Native DeepSpeed update with homogeneous physical batches and fixed logical batch.

    DeepSpeed still scales by the original GAS. Multiplying each group mean by
    its example count therefore preserves the original per-example loss sum.
    Only the final group reduces gradients and steps Adam/LR. Metrics reduce
    scalars, since ranks may have different numbers of homogeneous groups.
    """
    logical_steps = args.gradient_accumulation_steps
    timers('batch generator').start()
    groups = grouped_batches([next(data_iterator) for _ in range(logical_steps)], args)
    timers('batch generator').stop()
    initial_micro_steps = model.micro_steps
    weighted_loss = None
    weighted_metrics = {}
    try:
        for i, (original_micro, count, data) in enumerate(groups):
            last = i == len(groups) - 1
            model.set_gradient_accumulation_boundary(last)
            args._gram_microstep = original_micro
            timers('forward').start()
            loss, metrics = forward_step(neox_args=args, timers=timers,
                data_iterator=iter([data]), model=model, is_train=True,
                reference_model=reference_model)
            timers('forward').stop()
            timers('backward').start()
            backward_step(neox_args=args, timers=timers, optimizer=optimizer,
                          model=model, loss=loss * count)
            timers('backward').stop()
            share = count / logical_steps
            term = loss.detach() * share
            weighted_loss = term if weighted_loss is None else weighted_loss + term
            for key, value in metrics.items():
                term = value.detach() * share
                weighted_metrics[key] = weighted_metrics.get(key, 0) + term
            timers('optimizer').start()
            if last:
                model.module._gram_controller.prepare_step(model.device)
            model.step()
            if last:
                model.module._gram_controller.finish_step()
            timers('optimizer').stop()
        # micro_steps tracks consumed singleton examples, not grouped launches.
        model.micro_steps = initial_micro_steps + logical_steps
    finally:
        model.set_gradient_accumulation_boundary(None)
    result = {key: reduce_losses([value]).mean() for key, value in weighted_metrics.items()}
    result['lm_loss'] = reduce_losses([weighted_loss]).mean()
    return result, 0
