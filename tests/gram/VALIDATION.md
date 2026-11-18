# Packaged-branch validation

4 November 2026, CoreWeave CPU compute allocation (job 20966):

```text
python -m pytest --confcutdir=tests/gram -q tests/gram
73 passed, 2 skipped in 480.46s
```

The skipped tests require explicit opt-in and a compatible actual DeepSpeed
CPU wrapper environment. The passing tests cover routing, auxiliary spread,
nested roles, frozen optimizer state/decay, grouped accumulation, BF16 cast/add,
homogeneous mmap packing, exports and CPU distributed synchronization.

No new full-size GPU run was launched for packaging. This result does not
resolve the loss spikes/plateau in the historical experiments, establish native
NeoX/HF logit parity, or certify long-run behavior across DeepSpeed revisions.

## 18 November 2026: init, per-rank routing, gradient-norm logging

Changes on `gram/deep-ignorance-20261118`:

- Adapter layers use the model's init functions: `init_method` for the input projection and
  `output_layer_init_method` for the output projection (scaled by layer count with the default `scaled_normal`).
  Previously both used plain `normal(init_std)`, ~8x too large at 32 layers.
- The routing key includes the data-parallel rank, so each rank draws its own route.
- RoutedAdamW records per-role squared gradient norms each step; the training log prints the global gradient norm
  and per-role norms (pre-clipping) with their maxima since the previous log line, and writes them to TensorBoard
  under `gram/`.

Local CPU run (torch 2.x CPU, no DeepSpeed):

```text
python -m pytest --confcutdir=tests/gram -q tests/gram
79 passed, 2 skipped
```
