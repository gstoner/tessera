---
status: Active review snapshot
classification: Audit / Source-base review
authority: Summarizes the repository state after the May 2026 GA + EBM development burst
last_updated: 2026-05-17
---

# Tessera Source-Base Review — Clifford Algebra + Energy-Based Model Direction

This review captures the state of the source base after a substantial run of
development since the prior checkpoint. The main change is strategic: Tessera is
no longer only a tile-centric ML/HPC compiler with backend experiments. It now
has a second, sharper thesis:

> Tessera can make mathematical structure first-class in the compiler IR,
> starting with Clifford / geometric algebra and energy-based models.

The source tree now contains enough Python reference implementation, primitive
registry wiring, MLIR dialect scaffolding, lowering-pass bodies, backend
manifests, Apple GPU hooks, and conformance tests to treat the GA + EBM
direction as an active architecture track rather than a loose research note.

## Review Method

Reviewed on 2026-05-17 from `main` at commit `afe949a` (`Apple Back in place`).
The working tree was clean before this document was added.

Commands and checks used:

- `git log --oneline -n 25` to identify the recent development burst.
- `rg --files` and focused `find` scans over `python/`, `src/`, `docs/`, and
  `tests/`.
- Focused GA + EBM test sweep through the repo virtualenv:

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

Result: **473 passed in 7.36s**.

Note: the global `/opt/homebrew/bin/pytest` shim is stale on this machine and
points at a removed Python 3.13 interpreter. Use `.venv/bin/python -m pytest`
for local validation.

## Executive Assessment

The GA + EBM direction is coherent and unusually well documented. The core
decision records are in:

- [`docs/audit/ga_ebm_roadmap.md`](ga_ebm_roadmap.md)
- [`docs/audit/ga_scope_lock.md`](ga_scope_lock.md)
- [`docs/audit/ebm_scope_lock.md`](ebm_scope_lock.md)
- [`docs/spec/EBM_SPEC.md`](../spec/EBM_SPEC.md)

The roadmap records GA0 through GA11 and EBM0 through EBM8 as landed. That is
credible at the Python/reference, registry, dialect-scaffold, pass-body, and
test-contract layers. For GA specifically, Apple GPU fused-kernel coverage now
spans all 17 registered primitives. The remaining gap is integration maturity:
the GA kernels still need a normal Python/JIT → Clifford dialect → Apple
lowering → Tessera runtime → benchmark path, and EBM still needs backend
lowering plus quantitative performance validation.

In short: the source base has moved from "Tessera can target many backends" to
"Tessera can encode geometry and energy minimization as compiler-native
objects." That is a stronger and more differentiating direction.

## What Has Changed Since The Prior Checkpoint

Recent git history shows a concentrated sequence:

- `a1e2df9` — Clifford Algebra support for the Compiler
- `3bb4ef9` — C++ support for Clifford Algebra
- `6d82abf` — MLIR integration
- `3299393` / `afe949a` — Apple support brought back into place
- surrounding backend, datatype, JVP/VJP, and documentation updates

The codebase now includes:

- `python/tessera/ga/` — first-class Clifford / geometric algebra namespace.
- `python/tessera/ebm/` — first-class energy-based model namespace.
- `python/tessera/autodiff/geometric/` — parallel GA autodiff registry.
- `src/solvers/clifford/` — MLIR dialect, passes, driver, and lit fixtures.
- `src/solvers/ebm/` — MLIR dialect, passes, driver, and lit fixtures.
- GA and EBM entries in `primitive_coverage.py` and backend manifests.
- Focused conformance tests for rotation invariance, Lorentz invariance,
  RBM-style training, EBT-style inner-loop refinement, and bivector sampling.

## Clifford / GA Track

The GA track is currently the stronger of the two new surfaces.

Shipped Python surface:

- `Cl(p, q, r)` signatures with v1 locked to `Cl(3,0)` and `Cl(1,3)`.
- `Multivector` as a sibling tensor kind.
- Grade-aware constraints and annotations.
- Core operations: geometric product, grade projection, wedge, contraction,
  inner product, reverse, grade involution, conjugation, norm, exp/log, rotor
  construction, and rotor sandwich.
- Differential-form operations: Hodge star, exterior derivative, codifferential,
  vector derivative, and integration.
- Manifold helpers for Euclidean domains, spheres, and `SO(n)` stubs.
- Parallel multivector VJP/JVP registry.

Compiler surface:

- `tessera_clifford` dialect with 17 ops under `src/solvers/clifford/`.
- `ts-clifford-opt` driver.
- Passes for algebra annotation, rotor-sandwich folding, grade fusion, and
  product-table expansion.
- Product expansion uses compile-time Cayley tables and lowers rank-1 v1
  multivectors to tensor/arithmetic operations.

Backend surface:

- Backend manifest entries exist for all 17 GA primitives.
- Apple GPU coverage is fused for all 17 registered GA primitives. The final
  GA11 symbols cover exp/log, exterior derivative, vector derivative,
  codifferential, and integral; focused tests validate symbol export and
  Python-reference agreement.
- x86 / Apple CPU are presently reference-first paths; native batched C++
  kernels remain a next maturity step.

Important risk:

- The GA autodiff registry exists and per-op checks are strong, but full
  tape-based recording of arbitrary GA operation graphs is not yet the same
  maturity level as the conventional tensor autodiff path. This is acceptable
  for the current phase, but it should stay visible.

## EBM Track

The EBM track is broad and well staged. Its biggest strength is that it does
not depend on GA until manifold-aware sampling, so the Euclidean EBM surface can
stand on its own.

Shipped Python surface:

- `energy`, `inner_step`, `langevin_step`, `self_verify`, `decode_init`.
- Langevin, MALA, HMC, and Gibbs samplers via the RNG surface.
- Exact, Monte Carlo, and AIS partition-function estimators.
- Contrastive divergence, persistent CD, implicit score matching, and denoising
  score matching losses.
- Manifold-aware sphere and bivector Langevin sampling.

Compiler surface:

- `tessera_ebm` dialect with 6 core ops.
- `ts-ebm-opt` driver.
- Passes for canonicalization, energy/gradient fusion annotation,
  inner-loop checkpoint annotation, and candidate pipelining.

Important risk:

- EBM6 passes are currently annotation-layer transformations. The quantitative
  promises in the design, such as memory-traffic reduction from fused
  energy/gradient evaluation, require backend codegen and benchmark harnesses.
  The source tree is honest about this, and the review agrees.

## Documentation Health

Documentation is strong for the GA + EBM work. The roadmap is detailed,
acceptance-driven, and has useful "honest deviation" notes where the delivered
version differs from the original acceptance text.

The one weakness is discoverability. The top-level `README.md` still frames
Tessera primarily as a tile-centric deep learning and HPC compiler. That is
true, but it undersells the newer compiler-native mathematics direction. A
future edit should add a short "Research Directions" or "Mathematical IR
Surfaces" section pointing to the GA + EBM roadmap and EBM spec.

Recommended doc follow-ups:

1. Add a top-level README pointer to GA + EBM.
2. Add a compact Clifford spec under `docs/spec/`, parallel to `EBM_SPEC.md`.
3. Add a "native execution status" table that separates Python reference,
   MLIR/lit, manifest, and real hardware execution for each GA/EBM component.

## Testing Health

The focused GA + EBM unit suite is healthy: **473 passed**. The tests are not
just import checks; they cover algebraic identities, random rotor/SO(3)
agreement, Lorentz invariance, gradient checks, sampler properties, partition
estimators, EBM losses, and dialect wiring.

There are still three validation tiers to keep distinct:

- **Green now:** Python/reference behavior and source-tree wiring.
- **Partially green / environment-dependent:** MLIR 21 build and lit fixtures.
- **Future maturity:** native C++/MSL/CUDA/HIP execution benchmarks for GA/EBM
  kernels and EBM inner-loop performance claims.

## Architectural Direction

The new direction is worth leaning into.

Clifford algebra gives Tessera an IR-level way to represent the signature of
space: Euclidean `Cl(3,0)` for rotations and geometry, Minkowski `Cl(1,3)` for
Lorentz-invariant models. That is not a convenience wrapper; it changes what
the compiler can prove and preserve.

Energy-based models give Tessera an IR-level way to represent optimization and
sampling loops: candidate initialization, repeated refinement, energy/gradient
reuse, self-verification, partition estimation, and manifold-aware Langevin
dynamics. That lines up naturally with Tessera's existing strengths in tiling,
memory planning, scheduling, and backend lowering.

Together they create a clear research identity:

- Geometry is encoded in types and operations, not ad hoc tensor conventions.
- Energy minimization and sampling are scheduled compiler patterns, not only
  Python control flow.
- Manifold constraints are maintained by construction through grade projection,
  tangent projection, and retraction.
- Backend specialization can target algebra-specific kernels instead of
  generic dense tensor code.

## Near-Term Recommendations

1. **Make the direction visible at the top level.** Add a README section for
   compiler-native mathematical IR surfaces with links to GA + EBM docs.

2. **Write the Clifford spec.** `EBM_SPEC.md` exists and reads as normative.
   GA deserves the same kind of compact spec that covers signatures,
   multivector kind, op contracts, autodiff, lowering, and backend status.

3. **Separate status labels by layer.** Avoid one word like "landed" carrying
   Python reference, MLIR scaffold, pass bodies, manifest slots, and real
   hardware execution. The source tree already uses status labels; GA/EBM would
   benefit from a layer-by-layer table.

4. **Promote one end-to-end demo.** Choose one visible demo for each track:
   rotation-invariant point-cloud features for GA, and EBT-tiny inner-loop
   refinement for EBM. Keep them small, deterministic, and runnable in CI.

5. **Close the native execution gap deliberately.** The next meaningful
   engineering milestone is not more primitive breadth; it is one fully native,
   measured GA/EBM path from Python API to dialect to lowering to backend
   execution with a reproducible benchmark.

## Bottom Line

The repository has seen a fair amount of development, and the work has a
recognizable shape. The GA + EBM track is the strongest new narrative in the
tree: it differentiates Tessera from "another tensor compiler" and pushes it
toward a compiler for structured mathematical computation.

The right next move is to consolidate, not sprawl: make the direction easy to
find, specify Clifford as cleanly as EBM, keep the status layer honest, and pick
one or two end-to-end execution paths that prove the idea beyond reference
Python.
