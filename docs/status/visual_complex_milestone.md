# Visual Complex Analysis (M7) — milestone status (engineering-internal)

> **Engineering audit; not the public landing page.**  External
> readers, partners, and downstream tooling should land on
> [`docs/status/visual_complex.md`](visual_complex.md) (one-page
> public status) and follow links back here for the engineering claims.
>
> **One-page canonical status.** Update this page when something
> changes; everything else in the repo (READMEs, roadmap, audit) cites
> the *claims* below rather than restating them.
>
> **Last updated:** 2026-05-20 — E3 partial-ops uplift landed.  M7
> long-tail ops now have OP_SPECS rows, the audit walker reports
> ``runnable_reference`` (was ``partial``) for the 16 non-fused ops,
> and the backend manifest reserves ``status="planned"`` slots across
> apple_gpu / nvidia_sm80..120 / rocm with the target kernel dtype
> matrix (fp32 + fp16 + bf16) so Phase G / H / M7 follow-up have a
> stable hand-off point.  See ``docs/audit/partial_ops_uplift_plan.md``.
>
> **Dtype reading rule.** A ``planned`` row's dtype tuple is the
> **target kernel dtype matrix** — what the unbuilt native kernel
> will support — not what runs today.  The default/reference M7 path
> runs on CPU in fp32; four ops also have fused Apple GPU fp32 native
> kernels. fp16/bf16 entries land alongside the future kernels. Across
> the rest of the repo, see ``BackendKernelEntry.dtypes`` for the
> normative interpretation.
>
> **Previous update (2026-05-19):** M7 surface visible in
> `docs/audit/generated/support_table.md` (20 primitive rows, family
> `visual_complex`). Implementation + 94 focused tests had landed
> earlier; the 2026-05-19 milestone closed the audit-visibility gap.

## TL;DR

| Surface | Status | Coverage |
|---|---|---|
| **ℂ-arithmetic primitives** | ✅ Python reference + tests | 10 ops: `complex_add/mul/div/exp/log/sqrt/pow/conjugate/abs/arg` in `tessera.complex` |
| **Möbius transformation surface** | ✅ Python reference + tests | `mobius`, `mobius_from_three_points`, `cross_ratio`, `is_concyclic` |
| **Conformal geometry** | ✅ Python reference + tests | `stereographic`, `conformal_jacobian`, `conformal_energy_on_sphere` |
| **Wirtinger calculus** | ✅ Python reference + tests | `dz` (∂/∂z), `dbar` (∂/∂z̄), `laplacian_2d` (4 ∂² /∂z∂z̄) |
| **Cauchy-Riemann verification** | ✅ Decoration-time gate | `check_cauchy_riemann`; backs the `@analytic` and `@complex_jit` decorators |
| **`@complex_jit` symbolic frontend** | ✅ AST → symbolic-graph lowering | `python/tessera/compiler/complex_jit.py`; lowers a Python-source `f(z)` to a CR-verified symbolic graph |
| **Support-table visibility** | ✅ **landed 2026-05-19** | 20 primitive rows in `support_table.md` under family `visual_complex` |
| **Native (Apple GPU) lowering** | 🟢 4 fused + 16 planned slots | `complex_mul` / `complex_exp` / `mobius` / `stereographic` ship fused MSL kernels (fp32). The remaining 16 long-tail ops have `status="planned"` slots reserved in the backend manifest (fp32/fp16/bf16 target dtypes); execution lights up with the next M7 kernel sprint. |
| **NVIDIA / ROCm lowering** | 🟡 planned slots reserved | Every M7 op has `status="planned"` rows across nvidia_sm80/90/100/120 + rocm with `(fp32, fp16, bf16)` as the **kernel target** dtype matrix — these are what the unbuilt kernels will support, not what runs today. Promotion gated on Phase G / Phase H. See §9 of `docs/nvidia_cuda13_kernel_inventory.md` + §10 of `docs/rocm_mfma_kernel_inventory.md`. |
| **Today's execution path** | ✅ CPU reference default + 4 Apple GPU native kernels | The default/reference path runs via `tessera.complex.*` on CPU with fp32 precision. `complex_mul`, `complex_exp`, `mobius`, and `stereographic` also have fused Apple GPU fp32 kernels. fp16/bf16 are **not** runtime-supported yet — they're declared dtypes for future native kernels. |
| **E2E coverage classification** | ✅ no partial rows | E2E audit classifies 4 fused Apple-GPU ops as `complete` and 16 long-tail ops as `runnable_reference`; `complex_jit` remains a decorator/front-end surface rather than an op-counted primitive. |

## What's claimed

- **20 primitives** registered in `primitive_coverage.py` with category
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
  the 20 primitive rows render with `api=public` + `frontend=public` (source:
  `tessera.complex.*`), and partial/planned glyphs on
  the remaining axes — exactly the GA/EBM contract that lets the
  table read as honest signal rather than aspiration.

## What's NOT claimed

- **No native kernels for the 16 long-tail M7 ops.** Four Apple GPU
  kernels (`complex_mul`, `complex_exp`, `mobius`, `stereographic`)
  are fused today; the remaining long-tail ops run through the
  fp32 Python reference path until their planned backend slots are
  implemented.
- **No `tessera.complex` integration with the broader autodiff tape
  yet.** `@analytic` and `@complex_jit` verify CR at decoration time;
  the per-primitive VJPs / JVPs are not yet registered against
  `tessera.autodiff.vjp._VJPS` (so the registry's `vjp` axis reads
  `planned` for the M7 family).
- **No claim that planned fp16/bf16 rows execute today.** Long-tail
  M7 `planned` backend rows reserve future native-kernel dtypes; the
  current reference execution path is fp32-only.

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
| Primitive coverage registry (20 rows) | `python/tessera/compiler/primitive_coverage.py` — search `# ── M7:` |
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
