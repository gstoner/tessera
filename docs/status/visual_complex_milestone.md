# Visual Complex Analysis (M7) — milestone status

> **One-page canonical status.** Update this page when something
> changes; everything else in the repo (READMEs, roadmap, audit) cites
> the *claims* below rather than restating them.
>
> **Last updated:** 2026-05-19 — M7 surface visible in
> `docs/audit/generated/support_table.md` (22 rows, family
> `visual_complex`). Implementation + 94 focused tests had landed
> earlier; this milestone closes the visibility gap that left M7 out
> of the audit dashboard.

## TL;DR

| Surface | Status | Coverage |
|---|---|---|
| **ℂ-arithmetic primitives** | ✅ Python reference + tests | 10 ops: `complex_add/mul/div/exp/log/sqrt/pow/conjugate/abs/arg` in `tessera.complex` |
| **Möbius transformation surface** | ✅ Python reference + tests | `mobius`, `mobius_from_three_points`, `cross_ratio`, `is_concyclic` |
| **Conformal geometry** | ✅ Python reference + tests | `stereographic`, `conformal_jacobian`, `conformal_energy_on_sphere` |
| **Wirtinger calculus** | ✅ Python reference + tests | `dz` (∂/∂z), `dbar` (∂/∂z̄), `laplacian_2d` (4 ∂² /∂z∂z̄) |
| **Cauchy-Riemann verification** | ✅ Decoration-time gate | `check_cauchy_riemann`; backs the `@analytic` and `@complex_jit` decorators |
| **`@complex_jit` symbolic frontend** | ✅ AST → symbolic-graph lowering | `python/tessera/compiler/complex_jit.py`; lowers a Python-source `f(z)` to a CR-verified symbolic graph |
| **Support-table visibility** | ✅ **landed 2026-05-19** | 22 rows in `support_table.md` under family `visual_complex` |
| **Native (Apple GPU) lowering** | 🟡 deferred | M7 surface routes through the CPU/numpy reference path. Native (MSL kernel) lanes are the next gate; planned to ride the same `_apple_gpu_dispatch` + manifest infrastructure that GA/EBM use. |
| **Contract-axis completeness** | 🟡 partial | All 22 rows are at `status="partial"` in `primitive_coverage.py`. Math / shape / dtype / VJP / JVP / batching / transpose / sharding / lowering / kernel / tests axes need promotion before the M7 family can claim contract-complete. |

## What's claimed

- **22 primitives** registered in `primitive_coverage.py` with category
  `visual_complex`. Each carries a Needham reference and a one-line
  semantic note.
- **94 focused complex tests** pass in `tests/unit/` covering:
  - ℂ-arithmetic and Wirtinger derivatives (`test_complex_*.py`)
  - Möbius transformation algebra + cross-ratio invariance
  - Stereographic projection round-trip
  - Cauchy-Riemann checks on a curated holomorphic + non-holomorphic
    surface
  - Symbolic `@complex_jit` decoration-time lowering (and its
    rejection of non-holomorphic functions)
  - Runtime bridge tests verifying the symbolic graph re-evaluates to
    the numpy reference within float-precision tolerance
- **Per-axis visibility** in `docs/audit/generated/support_table.md`:
  the 22 rows render with `api=public` + `frontend=public` (source:
  `@complex_jit / tessera.complex.*`), and partial/planned glyphs on
  the remaining axes — exactly the GA/EBM contract that lets the
  table read as honest signal rather than aspiration.

## What's NOT claimed

- **No native (Apple GPU / NVIDIA / ROCm) MSL/PTX/AMDGCN kernels for
  the M7 family.** The current execution path is numpy-backed
  CPU reference. A future M7.1 milestone is expected to add fused
  MSL kernels (`complex_mul`, `complex_exp`, `mobius`,
  `stereographic`) on Apple GPU using the same manifest-backed
  dispatch the GA / EBM primitives use.
- **No `tessera.complex` integration with the broader autodiff tape
  yet.** `@analytic` and `@complex_jit` verify CR at decoration time;
  the per-primitive VJPs / JVPs are not yet registered against
  `tessera.autodiff.vjp._VJPS` (so the registry's `vjp` axis reads
  `planned` for the M7 family).
- **No M7 entries in `op_catalog.OP_SPECS`.** The audit picks them
  up via the curated `_M7_INVENTORY` set in
  `python/tessera/compiler/audit.py`, parallel to `_BENCH_INVENTORY`
  for GA/EBM. This is intentional — they're "callable today via
  `tessera.complex.*`" but not yet "registered as Graph IR ops."

## How to reproduce the claims

```bash
# Focused test sweep (94 tests, ~3s):
PYTHONPATH=python pytest tests/unit/test_complex_*.py \
    tests/unit/test_analytic_decorator.py \
    tests/unit/test_complex_jit_*.py -v

# Support-table drift gate (now includes M7 rows):
PYTHONPATH=python python -m tessera.compiler.audit support_table --check

# Inspect the M7 rows directly:
grep "^\| .*visual_complex" docs/audit/generated/support_table.md
```

## Where to look in the code

| Surface | Module |
|---|---|
| Python reference + decorators | `python/tessera/complex.py` |
| Symbolic `@complex_jit` frontend | `python/tessera/compiler/complex_jit.py` |
| Primitive coverage registry (22 rows) | `python/tessera/compiler/primitive_coverage.py` — search `# ── M7:` |
| Support-table inventory | `python/tessera/compiler/audit.py` — `_M7_INVENTORY` |
| Conformal energy helpers | `python/tessera/conformal_advanced.py` |
| Tests | `tests/unit/test_complex_*.py`, `tests/unit/test_analytic_decorator.py`, `tests/unit/test_complex_jit_*.py` |

## Roadmap (the next M7.x milestones)

- **M7.1 — autodiff registration.** Register VJP / JVP for the M7
  primitives in `tessera.autodiff.vjp._VJPS` / `_JVPS` so the audit
  reports `vjp=complete` / `jvp=complete` instead of `planned`.
- **M7.2 — Apple GPU fused MSL kernels.** Mirror the GA/EBM pattern:
  fused MSL for `complex_mul`, `complex_exp`, `mobius`,
  `stereographic`, `cross_ratio` plus a workload chain (e.g., a
  conformal-map composite). Add to
  `_CLIFFORD_APPLE_GPU_FUSED`-style manifest entries.
- **M7.3 — Hyperbolic-geometry composite workload.** Builds on
  `mobius` + `stereographic` to demonstrate a non-Euclidean ML
  application (e.g., embedding-on-Poincaré-disk training step) as
  a third workload row alongside `ga_feature_pipeline` and
  `ebt_tiny_refinement`.

These ride the same manifest + jit-bridge infrastructure GA / EBM
already use; no new compiler machinery should be needed.
