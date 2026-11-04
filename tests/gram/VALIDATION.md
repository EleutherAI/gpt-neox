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
