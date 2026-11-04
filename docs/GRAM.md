# Gradient-routed auxiliary modules (research implementation)

This opt-in implementation adds small GELU MLP branches to each Transformer
MLP and selects parameter-gradient contributions using supplied data labels.
It includes the single-adapter design and an experimental nested two-adapter
extension. It is based on NeoX commit `beb61c04bf6812a236892162f3fdf387d1a47e6f`,
the training base used for the Deep Ignorance experiments, rather than a port
to the latest NeoX main branch.

**Validation status:** short CPU/distributed checks cover routing, optimizer
state, accumulation, exports and data packing. Historical large runs showed
loss spikes and a plateau; their cause is unresolved. This is a reviewable
research implementation, not a claim of validated large-scale convergence or
an exact reproduction of every experiment in the GRAM paper.

## Read the implementation

- `megatron/gram.py`: `FlaggedDataset`, `RoutingController`, `AuxiliaryMLP`,
  role-separated `RoutedAdamW`, removable prefixes and grouped microbatches.
- `megatron/training.py`: install before optimizer construction, select routes
  before forward, and synchronize role activity at optimizer boundaries.
- `megatron/model/transformer.py`: optional allocation of auxiliary channels
  from the original MLP width. The default instead keeps the full core width.
- `megatron/data/gram_packed_dataset.py`: optional homogeneous packing.
- `megatron/bf16_accumulation.py`: optional performance optimizations. Leave
  these disabled for a simpler stock-accumulation diagnostic baseline.
- `tests/gram/`: focused tests that import the small GRAM module directly.

## Single-adapter routing

The core includes embeddings, attention, normalization, unembedding and core
MLPs. The adapter contains only its additional per-layer MLP parameters.

| Sample | Forward | Parameter updates |
|---|---|---|
| Clean | Core; adapter additionally enabled with probability `p_cr` | Core and any enabled adapter |
| Flagged | Core + adapter | Adapter; core additionally updated with probability `p_as` |

`gram_core_robustness` is `p_cr`; `gram_aux_spread` is `p_as`. Strict isolation
uses `p_as=0`. The paper's realistic-data setting uses `p_as=0.5`, `p_cr=0.2`;
these are experimental settings, not universally optimal hyperparameters.
Draws are deterministic functions of seed, iteration and logical microstep.
Parameter-gradient masks preserve derivatives through activations. A role
inactive across every data-parallel rank also skips Adam state and decay.

For nested routing, labels are clean / outer-only / inner. Forward passes use
valid prefixes: core, core+outer, core+outer+inner. With spread zero, each
flagged category updates only its assigned adapter. Nonzero spread also
updates less restrictive roles. This hierarchy is an extension, not the
paper's main independent-domain design.

## Labels and packing

`gram_labels_path` is a one-dimensional binary NumPy array (`allow_pickle=False`).
`gram_label_granularity` selects labels by final sample index or source document
index. Document labels require a single GPT2 dataset. With ordinary packed
samples, any flagged document assigns the entire input and next-token target
sequence to the flagged route. This can route clean fragments as flagged.

To avoid mixed-label samples, build index-only streams:

```bash
python tools/gram/prepare_homogeneous_packing.py \
  --repo . --config production.json --output /path/to/new/packing
```

Set `gram_homogeneous_path` to the completed directory. The builder uses RAM
for deterministic shuffling and sequential array writes, audits 10,000 sample
joins and writes a completion marker. Each sample's inputs and target belong
to one class. The original token proportions are preserved up to rounding,
but document neighbors, attention contexts and batch composition change.
This is not per-token routing on the original mixed batches. Index construction
can require several GiB of RAM and should run in a compute allocation.

## Supported configuration

See `configs/gram/example.yml` for an illustrative small configuration. Supply
your own tokenizer, corpus and document-label paths before using it.

- Normal language-model training, GELU MLP, TP=1 and PP=0.
- FP32 or BF16; ZeRO stage 0/1; Adam/AdamW role layout version 2 (single)
  or 3 (nested). Explicit optimizer parameters and the ZeRO opt-in are required.
- Logical microbatch size 1. Optional grouping combines identical routing
  states while preserving the original logical batch size and mean loss.
- No Transformer Engine fused MLP/attention, MoE, optimizer offload or muP.
- Keep core width with `gram_absorb_auxiliary_width: false`. Setting it true
  removes auxiliary widths from the core; this changes core capacity.
- BF16 cast-and-add and early clearing are separate opt-ins, disabled by default.
  Their compatibility depends on DeepSpeed's BF16 optimizer implementation.

## Evaluation, exports, and resume

`gram_eval_level` fixes the enabled prefix for all evaluation inputs, with no
benchmark-dependent routing. For single adapter use 0 or 1; nested also has 2.
The all-adapter model is not a separately trained unfiltered dense control.

```bash
python tools/gram/export_core.py trusted_rank_checkpoint.pt core_only.pt --level 0
```

Exports contain model weights and explicit configuration overrides. They omit
optimizer/RNG state and cannot resume the original training run. Full training
resume requires the complete distributed checkpoint and matching routing settings.
Only load checkpoint files you trust.

## Tests

Install PyTorch, NumPy and pytest, then run from this repository root:

```bash
python -m pytest --confcutdir=tests/gram -q tests/gram
```

The optional actual-DeepSpeed CPU wrapper checks require a compatible
DeeperSpeed installation and explicit `GRAM_TEST_DEEPSPEED=1 DS_ACCELERATOR=cpu`.
Passing the short tests does not resolve the historical convergence issue.

The branch intentionally excludes cluster-specific launch scripts, data,
model checkpoints, compiled extensions and unrelated precomputed-window replay.
