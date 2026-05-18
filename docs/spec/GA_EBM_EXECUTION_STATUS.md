---
status: Normative (status ledger)
classification: Spec
authority: Separates Python reference, MLIR/lit, manifest, and native execution status for GA and EBM
last_updated: 2026-05-17
---

# GA + EBM Execution Status

This ledger separates implementation layers for Tessera's geometric algebra
and energy-based model surfaces. Use it when a component is described as
"landed" so that Python reference behavior, compiler/lit coverage, backend
manifest coverage, and real hardware execution are not conflated.

Status labels follow [`docs/README.md`](../README.md):

- `implemented` — active source exists and has unit or lit coverage.
- `lit-testable` — compiler behavior has MLIR lit or contract fixtures; native
  execution is not implied.
- `mock-runtime` — deterministic Python or CPU/mock fallback exists.
- `hardware-runtime` — native execution is wired for a backend and has a
  concrete build/test path.
- `scaffolded` — API or directory shape exists, but behavior is incomplete.
- `planned` — design direction only.

## Summary Table

| Component | Python reference | MLIR / lit | Backend manifest | Native execution |
|-----------|------------------|------------|------------------|------------------|
| GA signature and multivector values | implemented | n/a | n/a | mock-runtime via Python/NumPy reference |
| GA primitive ops and calculus | implemented | lit-testable through `tessera_clifford` op fixtures and lowering fixtures | implemented for all registered `clifford_*` primitives | hardware-runtime for 17/17 registered Apple GPU fused GA kernels, benchmarked by `benchmark_ga_ebm.py --ci`; x86 and Apple CPU remain reference-first; NVIDIA/ROCm planned |
| GA geometric autodiff | implemented | n/a | n/a | mock-runtime via Python reference checks |
| GA dialect and lowering passes | n/a | implemented / lit-testable | n/a | native execution only where a backend consumes the lowered artifacts |
| EBM energy primitives | implemented | lit-testable through `tessera_ebm` parse/canonicalize fixtures | implemented for registered `ebm_*` primitives where manifests apply | Apple GPU hardware-runtime for quadratic `ebm_energy`; arbitrary user `energy_fn` lowering remains planned |
| EBM samplers and partition estimators | implemented | scaffolded for future lowering | implemented / partial depending on primitive | Apple GPU hardware-runtime for `ebm_langevin_step`, `ebm_bivector_langevin`, `ebm_sphere_langevin`, and **`ebm_partition_exact`** (single-dispatch stable-logsumexp MSL kernel landed 2026-05-17); `ebm_partition_monte_carlo` and `ebm_partition_ais` remain Python-only |
| EBM losses | implemented | n/a | n/a | mock-runtime through tensor loss/autodiff reference path |
| EBM dialect and annotation passes | n/a | implemented / lit-testable | n/a | no standalone hardware-runtime claim; backend codegen required |
| EBM inner-loop and decode primitives | implemented | lit-testable through EBM dialect fixtures | implemented for registered `ebm_*` primitives | Apple GPU hardware-runtime for `ebm_inner_step`, `ebm_refinement`, and `ebm_decode_init` |
| EBM self-verification | implemented | lit-testable through EBM dialect fixtures | implemented for registered `ebm_*` primitives | Apple GPU hardware-runtime for hard-argmin `ebm_self_verify`; soft-min remains planned |
| EBM manifold-aware GA sampling | implemented | represented through EBM/Clifford dialect surface | implemented / partial | Apple GPU hardware-runtime for `ebm_bivector_langevin` and `ebm_sphere_langevin` |
| GA/EBM composite workloads | implemented in benchmark driver | n/a | uses manifest-resolved native symbols where available | `ga_feature_pipeline`, `ebt_tiny_refinement`, and opt-in `--ebt-sweep` emit Apple GPU and Python-reference benchmark rows |

## Layer Notes

### Python Reference

The Python reference surface is the correctness source for both tracks:

- `python/tessera/ga/`
- `python/tessera/ebm/`
- `python/tessera/autodiff/geometric/`

These paths are covered by focused unit and conformance tests for algebraic
identities, rotor behavior, Lorentz invariance, gradient checks, Langevin /
MCMC sampling, partition estimators, losses, and small EBM demos.

### MLIR / Lit

The compiler-facing surfaces are:

- `src/solvers/clifford/`
- `src/solvers/ebm/`

Clifford has parse/print fixtures and pass fixtures for algebra annotation,
rotor-sandwich folding, grade fusion, and product-table expansion.

EBM has parse/print fixtures and pass fixtures for canonicalization,
energy-gradient fusion annotation, inner-loop checkpoint annotation, and
candidate-pipeline annotation.

### Backend Manifest

The backend manifest records intended target coverage and dtype support. A
manifest entry is not by itself a native execution claim. It is a contract that
the compiler and audit tooling can inspect.

For GA, all registered Clifford primitives have manifest coverage. Apple GPU
fused entries now cover all 17 registered GA primitives; the manifest test gate
asserts `FUSED_APPLE_GPU_OPS == EXPECTED_CLIFFORD_OPS` and
`PLANNED_APPLE_GPU_OPS == set()`. x86 and Apple CPU are reference-first for
this track; NVIDIA and ROCm remain planned for GA.

For EBM, manifests should be interpreted around the underlying primitive class.
Apple GPU fused entries currently cover nine benchmarked native ops:
`ebm_inner_step`, `ebm_refinement`, `ebm_langevin_step`, `ebm_decode_init`,
`ebm_bivector_langevin`, `ebm_sphere_langevin`, `ebm_self_verify`, and the
quadratic specialization of `ebm_energy`, plus `ebm_partition_exact` as a
single-dispatch stable-logsumexp kernel over precomputed energies.

### Native Execution

Native execution means an operation can run through a concrete backend build
and test path, not merely that it has a Python reference implementation or a
Target IR artifact.

Current high-confidence native claim:

- Clifford Apple GPU fused kernels for all 17 registered GA primitives.
- EBM Apple GPU fused kernels for nine registered native rows:
  `inner_step`, `refinement`, `langevin_step`, `decode_init`,
  `bivector_langevin`, `sphere_langevin`, hard-argmin `self_verify`, and
  quadratic `energy`, plus stable-logsumexp `partition_exact`.
- Composite workload benchmark rows for `ga_feature_pipeline`,
  `ebt_tiny_refinement`, and opt-in `--ebt-sweep`, each paired with
  Python-reference rows.

Current non-claims:

- Arbitrary user-defined EBM energy functions do not yet lower to native MSL;
  the current native `ebm_energy` row is the quadratic specialization.
- `ebm_partition_monte_carlo` and `ebm_partition_ais` remain Python-reference
  estimators; only exact partition over precomputed energies has a native
  Apple GPU kernel.
- EBM checkpointing and candidate pipelining beyond the measured
  `ebm_refinement` kernel are annotation-layer compiler transformations until a
  backend consumes them.
- NVIDIA and ROCm GA/EBM execution is planned, not current hardware-runtime.

## Validation Command

Use the repo virtualenv for the current GA + EBM native health check:

```bash
.venv/bin/python benchmarks/apple_gpu/benchmark_ga_ebm.py --ci
```

On Apple Silicon this emits 17 GA Apple GPU rows, eight native EBM Apple GPU
rows, Python-reference EBM rows for comparison, and the GA/EBM workload rows.
On non-Darwin hosts it records `skipped_apple_gpu` and still emits the
Python-reference rows.

Use the focused unit suite for GA + EBM API and dialect validation:

```bash
.venv/bin/python -m pytest \
  tests/unit/test_ga_namespace.py \
  tests/unit/test_ga_signature.py \
  tests/unit/test_ga_multivector.py \
  tests/unit/test_ga_ops.py \
  tests/unit/test_ga_calculus.py \
  tests/unit/test_ga_autodiff.py \
  tests/unit/test_ga_backend_manifest.py \
  tests/unit/test_ga_conformance.py \
  tests/unit/test_ebm_namespace.py \
  tests/unit/test_ebm_primitives.py \
  tests/unit/test_ebm_samplers.py \
  tests/unit/test_ebm_partition.py \
  tests/unit/test_ebm_losses.py \
  tests/unit/test_ebm_geo_sampling.py \
  tests/unit/test_ebm_conformance.py \
  tests/unit/test_clifford_dialect_wiring.py -q
```
