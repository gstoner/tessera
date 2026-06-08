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
- **GA close-out tranche (2026-06-08):** `clifford_rotor_sandwich` (the GA
  rotation-application op) joined the shim + autodiff + apple_gpu envelope
  (VJP/JVP composed from the geometric-product + reverse rules). The
  planned-entry coverage path now consults the live `_VJPS`/`_JVPS` registries,
  so GA/EBM rows reflect registered autodiff (**GA 10 complete / 7 planned**;
  the 7 = `exp`/`log` autodiff-pending + 5 unimplemented differential ops). The
  10 EBM callable/RNG ops are reclassified `vjp/jvp=not_applicable` (they take an
  `energy_fn` callable / RNG key — not flat tape ops). **Cl(1,3) signature gate
  (gap #4):** the Multivector front-end allow-list is `{Cl(3,0), Cl(1,3)}` but
  only Cl(3,0) has kernels; `@clifford_jit` now refuses a non-Cl(3,0) call at the
  boundary with `CLIFFORD_UNSUPPORTED_SIGNATURE` (Decision #21) instead of
  silently running the numpy reference — the plain `tessera.ga.*` lane stays
  available for Cl(1,3). **Still open:** Apple-CPU GA/EBM native kernels, EBM
  training losses (CD/PCD/ISM/DSM), GA/EBM cross-op fusion, and `exp`/`log`
  autodiff.
- **EBM training losses on Metal (gap #5, 2026-06-08):** the four EBM training
  losses — contrastive_divergence / persistent_cd / implicit_score_matching /
  denoising_score_matching (plus the explicit score_matching) — now have a
  kernel path. Dedicated MPSGraph reduction kernels
  (`tessera_apple_gpu_ebm_{energy_diff_mean,half_mse,ism,dsm}_f32`) compute the
  `reduction="mean"` case on GPU (sum/none fall back to the numpy reference);
  the ops are on the canonical `tessera.ops` surface + op_catalog
  (`tessera.loss.*`) + apple_gpu envelope, so `@jit(target="apple_gpu")` reports
  `execution_mode="metal_runtime"`. All five already had VJP+JVP. **Still open:**
  Apple-CPU GA/EBM native kernels, GA/EBM cross-op fusion, `exp`/`log` GA
  autodiff, and a pre-existing bf16 `control_for` lowering bug
  (iter_args≠results) tracked separately.
- **GA cross-op fusion (gap #6, 2026-06-08):** the canonical rotor-invariant
  `norm(rotor_sandwich(R, x))` now fuses into a single
  `clifford_rotor_sandwich_norm` dispatch. A fused MSL kernel
  (`tessera_apple_gpu_clifford_rotor_sandwich_norm_cl30_f32`) keeps the two
  geometric products in registers and emits only the scalar norm — no
  intermediate-multivector global-memory round-trip. `ga.rotor_sandwich_norm` is
  the direct entry; a `@clifford_jit` decorator fusion pass
  (`_fuse_rotor_sandwich_norm`) collapses the AST-lowered 2-op chain to the fused
  op when the intermediate is consumed exactly once (structural
  `lower_function_to_ir` stays unfused; fusion is a separate pass). Wired through
  the bridge manifest (`_CLIFFORD_APPLE_GPU_FUSED` + `_CLIFFORD_FUSION_OPS`, kept
  out of the 17 `_CLIFFORD_PRIMITIVES`). **Of the original 6 GA/EBM gaps, #1/#2/#5/#6
  are closed** for the shipped surface; remaining: Apple-CPU GA/EBM native
  kernels (#3), Cl(1,3) kernels (#4 gated with a diagnostic), and `exp`/`log` GA
  autodiff.
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
