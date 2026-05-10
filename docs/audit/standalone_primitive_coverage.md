---
status: Informative
classification: Audit Dashboard
authority: Companion dashboard for `docs/audit/execution_roadmap.md` S1
last_updated: 2026-05-10
---

# Standalone Primitive Coverage

This dashboard tracks Tessera-native compiler primitive completeness. PyTorch,
JAX, and Flax are reference vocabularies only; they are not runtime
dependencies and do not define Tessera semantics.

The source of truth for this dashboard is
`python/tessera/compiler/primitive_coverage.py`. `op_catalog.py` remains the
source of truth for currently accepted operators; this dashboard can include
planned primitives without falsely marking them as supported.

## Contract Axes

Every primitive is tracked across these contract fields:

- mathematical semantics
- shape rule
- dtype/layout rule
- VJP
- JVP
- batching/vectorization rule
- transpose rule
- sharding rule
- masking/effect behavior
- lowering rule
- backend kernel
- tests

## Milestone Groups

| Group | Purpose | Example primitives |
|-------|---------|--------------------|
| Tensor algebra | Baseline model graph expressivity | `reshape`, `dynamic_slice`, `dynamic_update_slice`, `cat`, `stack` |
| Indexing | Functional updates and retrieval | `scatter_add`, `scatter_reduce`, `top_k`, `sort`, `index_update` |
| Scalar math | Activation, loss, and schedule breadth | `exp`, `log`, `sqrt`, `rsqrt`, `pow`, `erf` |
| Control flow | Recurrent and dynamic models | `scan`, `associative_scan`, `while_loop`, `cond`, `switch` |
| State trees | Native model/state containers | `tree_flatten`, `tree_map`, `state_filter` |
| RNG | Reproducible stochastic compilation | `rng_key`, `rng_split`, `rng_fold_in`, `rng_bernoulli` |
| Sharding | Compiler-visible SPMD placement | `shard_map`, `named_sharding` |
| Model layers | Standalone model authoring | `linear_general`, `conv_transpose`, `group_norm`, `gru_cell` |
| Memory | Titans/Atlas-style learned memory | `memory_read`, `memory_write`, `memory_evict` |

## Model-Family Coverage Tags

The registry tags primitives by the model families they unblock:

- diffusion / DiT / U-Net
- RNN / xLSTM
- Mamba / SSM
- Hyena / FNet / spectral models
- Linformer / cosFormer
- Griffin / Megalodon
- Titans / Atlas memory
- JEPA

## Current S1 Result

S1 is complete when the registry and tests exist, not when every primitive is
implemented. The current result is intentionally a mixed dashboard:

- Existing Tessera operators are imported from `OP_SPECS` as partial coverage
  entries.
- Missing standalone compiler primitives are planned entries.
- Missing contract axes remain visible until each primitive has semantics,
  transform rules, lowering, backend coverage, and tests.

Generate the full table programmatically with:

```python
from tessera.compiler.primitive_coverage import render_markdown

print(render_markdown())
```
