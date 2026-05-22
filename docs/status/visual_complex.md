# Visual Complex / Conformal — public status

> **One-page public status.**  External readers, partners, and
> downstream tooling should land here.  The engineering-internal
> milestone audit lives at
> [`docs/status/visual_complex_milestone.md`](visual_complex_milestone.md)
> — cite *this* page in talks and READMEs; cite the milestone for
> the engineering claims behind each line.
>
> **Last updated:** 2026-05-22

## What is the Visual Complex lane?

The **Visual Complex** lane is Tessera's compiler surface for complex-
analytic, Möbius, and conformal-geometry computation — the M7 ship
landed in May 2026.  It provides:

* A complex-number arithmetic surface (`tessera.complex.*`).
* A Möbius / Wirtinger / conformal-geometry helper library.
* A **Cauchy-Riemann-verified `@complex_jit` decorator** that lowers
  a Python `f(z)` to a CR-checked symbolic graph at decoration time.
* Registry rows in
  [`primitive_coverage.py`](../../python/tessera/compiler/primitive_coverage.py)
  with `family = "visual_complex"` — visible in the generated support
  table.

## What you can do today

| Use case                          | Path                                                 |
|-----------------------------------|------------------------------------------------------|
| ℂ-arithmetic on a numpy backend   | `tessera.complex.*` (add, mul, div, exp, log, sqrt, pow, conjugate, abs, arg) |
| Möbius / fractional-linear        | `tessera.complex.mobius`, `mobius_from_three_points`, `cross_ratio`, `is_concyclic` |
| Conformal mapping                 | `tessera.complex.stereographic`, `conformal_jacobian`, `conformal_energy_on_sphere` |
| Wirtinger derivatives             | `tessera.complex.dz`, `dbar`, `laplacian_2d` |
| CR verification of a holomorphic `f(z)` | `@analytic` (decoration-time check) |
| CR-verified symbolic lowering     | `@complex_jit(target="apple_gpu")` — runs through `compiler/complex_jit.py` |
| Native Apple GPU fast paths       | 4 fused MSL kernels: `complex_mul`, `complex_exp`, `mobius`, `stereographic` (fp32) |

## Status at a glance

| Surface                          | Status               | Notes                                  |
|----------------------------------|----------------------|----------------------------------------|
| ℂ-arithmetic primitives          | ✅ shipped            | 10 ops, Python reference + 94 focused tests |
| Möbius / conformal helpers       | ✅ shipped            | full surface; cross-ratio invariance tested |
| `@complex_jit` symbolic frontend | ✅ shipped            | AST → CR-verified symbolic graph at decoration time |
| Apple GPU fused kernels          | ✅ 4 / 20             | `complex_mul`, `complex_exp`, `mobius`, `stereographic` |
| Apple GPU planned (16 long-tail) | 🟡 manifest reserved  | `status="planned"` slots with fp32/fp16/bf16 dtype matrix |
| NVIDIA / ROCm                    | 🟡 manifest reserved  | `status="planned"` per Phase G / H pre-work |
| Support-table visibility         | ✅ shipped            | 22 rows under family `visual_complex` in `docs/audit/generated/support_table.md` |
| Public benchmark                 | 🟡 indirect           | covered by `benchmarks/visual_complex_core` (cross-lane GA × EBM library benchmark) |
| Drift gate                       | ✅ shipped            | `tessera.compiler.audit support_table --check` (covers M7 rows) |

> **Dtype caveat.**  ``planned`` rows declare the *target kernel*
> dtype matrix (fp32 + fp16 + bf16) — that's what the unbuilt native
> kernel will support, not what runs today.  The reference path is
> fp32-only via Python.  The four fused Apple GPU kernels also run
> fp32 today; fp16 / bf16 land alongside the future kernels.

## Try it

```bash
# Focused test sweep (94 tests, ~3 s):
PYTHONPATH=python pytest tests/unit/test_complex_*.py \
    tests/unit/test_analytic_decorator.py \
    tests/unit/test_complex_jit_*.py -v

# Generated support-table drift gate (covers all 22 M7 rows):
PYTHONPATH=python python -m tessera.compiler.audit support_table --check

# Cross-lane (GA × EBM) library benchmark (uses Visual Complex ops):
PYTHONPATH=.:python python benchmarks/visual_complex_core/core.py \
    --reps 5 --output /tmp/vc.json
```

## What this page does NOT claim

* **No claim that the 16 long-tail M7 ops have native kernels yet.**
  They run through Python reference today; their planned backend
  slots are documented in the milestone doc.
* **No claim of `tessera.autodiff` integration.**  CR is checked at
  decoration time via `@analytic` and `@complex_jit`; per-primitive
  VJPs / JVPs aren't yet wired into `tessera.autodiff.vjp._VJPS`.
* **No claim of NVIDIA or ROCm execution.**  Backend rows reserve
  planned slots only — paired with Phase G / H hardware enablement
  per
  [`docs/audit/nvidia_rocm_execute_and_compare_plan.md`](../audit/nvidia_rocm_execute_and_compare_plan.md).

## Where to look in the code

| Surface                          | Module                                            |
|----------------------------------|---------------------------------------------------|
| Python reference + decorators    | `python/tessera/complex.py`                        |
| Symbolic `@complex_jit` frontend | `python/tessera/compiler/complex_jit.py`           |
| Primitive coverage (22 rows)     | `python/tessera/compiler/primitive_coverage.py` — search `# ── M7:` |
| Backend manifest entries         | `python/tessera/compiler/backend_manifest.py`      |
| Conformal helpers                | `python/tessera/conformal_advanced.py`             |
| Tests                            | `tests/unit/test_complex_*.py`, `test_analytic_decorator.py`, `test_complex_jit_*.py` |

## Related status pages

* [`docs/status/visual_complex_milestone.md`](visual_complex_milestone.md)
  — engineering-internal milestone status (the page this one
  summarises).
* [`docs/status/ga_ebm_milestone.md`](ga_ebm_milestone.md) — GA + EBM
  milestone; the cross-lane visual-complex benchmark composes both.
* [`docs/status/apple_release_gate.md`](apple_release_gate.md) — Apple
  release policy; covers the 4 fused M7 kernels.
