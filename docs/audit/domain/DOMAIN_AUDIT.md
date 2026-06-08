# Domain Audit

This document consolidates GA/EBM, attention, CorrDiff/SciML, sharding, and
autodiff domain audit material.

## Finished

- GA v1 scope is locked around the selected Clifford signatures and sibling
  multivector model.
- EBM scope decisions and archive disposition are locked.
- GA/EBM Python surfaces and Apple-specialized runtime lanes are substantially
  built.
- **GA/EBM lane unification (2026-06-08):** the GA Multivector lane
  (`tessera.ga.*`) and the tensor-clean EBM subset (`tessera.ebm.*`) are now
  projected onto the canonical `tessera.ops` surface via flat-array shims
  (`tessera.ops.clifford_*` — 10 ops; `tessera.ops.ebm_*` — 4 ops:
  energy_quadratic / self_verify / refinement / inner_step). This closes the two
  long-standing GA/EBM gaps: (#1) the ops flow through the autodiff tape with
  closed-form VJP+JVP rules (validated vs finite-difference), and (#2)
  `@jit(target="apple_gpu")` routes them to the cl30 / EBM MSL kernels and
  classifies them `execution_mode="metal_runtime"` (envelope-gated in
  `runtime.py` + `driver.py`). The authoritative `category="geometric_algebra"`
  / `category="ebm"` coverage rows are preserved (op_catalog OpSpecs feed IR
  emission but the OP_SPECS import skips the registry-owned names). Callable/RNG
  EBM ops (energy, partition_function*, langevin_step, decode_init) intentionally
  stay on `tessera.ebm` — they cannot be flat `tessera.ops` ops.
- Attention variants, MLA, speculative, KV-cache, and related surfaces have
  reference/compiler-facing implementations.
- CorrDiff analysis clarified compiler vs library/runtime ownership.
- Sharding partial audit classified open buckets instead of leaving a vague
  "distributed is partial" label.

## Still Open

- Domain roadmaps are not status authorities; generated dashboards must decide
  current support.
- Domain features that imply hardware kernels remain blocked on backend proof.
- GA/EBM/attention claims need to stay separated into Python reference,
  compiler lowering, Apple-specialized runtime, and non-Apple hardware proof.
- Some sharding/autodiff/domain crosscuts remain tied to long-tail batching,
  transpose, and sharding closure.

## Next Work

1. Keep domain README/status references pointed at generated dashboards.
2. Promote domain feature claims only when coverage and backend dashboards agree.
3. Continue pruning old roadmap prose into specific compiler/backend work items.
4. Use Apple-specialized success as evidence for Apple, not as a claim about
   NVIDIA/ROCm.

## Source Material Consolidated

- `archive/ga_ebm_roadmap.md`
- `archive/ga_scope_lock.md`
- `archive/ebm_scope_lock.md`
- `archive/ga6_autodiff_plan.md`
- `archive/attention_variants_plan.md`
- `archive/corrdiff_compiler_split_evaluation.md`
- `archive/sharding_partial_audit.md`
- `archive/source_base_review_2026_05_17.md`
