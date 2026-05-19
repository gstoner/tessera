---
status: Active planning snapshot
classification: Audit / Compiler improvement plan
authority: Reviews the source base and proposes the next compiler milestones
last_updated: 2026-05-18 (post-review revision)
---

# Tessera Compiler Improvement Milestone Plan - May 2026

This review looks at the compiler as a source base, not just as a set of
roadmap claims. It cross-checks the public API, Python compiler object model,
MLIR/lit scaffolding, runtime/codegen paths, tests, and the newer
Clifford/EBM work. It also assesses where ideas from Tristan Needham's
*Visual Complex Analysis* can usefully sharpen Tessera's geometry-aware
compiler direction.

## Executive assessment

Tessera has a credible compiler spine:

```text
Python API and textual frontend
  -> Graph IR
  -> Schedule IR
  -> Tile IR
  -> Target IR / backend artifact
  -> CPU, Apple CPU/GPU, or artifact-only backend paths
```

The strongest implemented path is still the Python object-model compiler:
`GraphIRModule` is the canonical source, then Schedule, Tile, and Target
objects are constructed and verified before textual artifacts are emitted.
This path is hardware-free, deterministic, and well covered by unit tests.

The source base has also added a second, more differentiated compiler theme:
mathematical structure as a compiler-visible concept. The Clifford/GA and EBM
tracks now have Python namespaces, specs, backend manifests, Apple GPU runtime
surface, benchmark proof bits, and a constrained `@clifford_jit` vertical
slice. That is more than a research note; it is an active architecture lane.

The main risk is layer ambiguity. Several features are "supported" at one
layer but not another:

- Python reference and CPU fallback can execute many ops.
- MLIR/lit fixtures verify many pass and dialect contracts.
- NVIDIA, ROCm, TPU, Metalium, Cerebras, and Rubin CPX are mostly target
  artifact or scaffold surfaces unless hardware-gated validation is run.
- Apple GPU is the most mature native non-CPU path, especially for GA/EBM and
  selected tensor ops.
- General `@tessera.jit` and `@clifford_jit` are not yet one unified frontend
  path.

The next compiler work should consolidate the vertical slices before adding
more breadth. The useful goal is not "more primitives"; it is "for each
important primitive family, the repo can show a complete and honest route from
frontend source to IR layers to codegen/runtime, with tests that prove exactly
which layers are real."

This plan is therefore a consolidation plan. It should reuse the machinery
already in the tree rather than invent parallel status, proof, or audit
records. The main existing sources are `primitive_coverage.py`,
`backend_manifest.py::BackendKernelEntry`, `capabilities.py`,
`op_catalog.py`, `JitBridgeRoute`, and the checked-in coverage dashboards.

## Recent landings (2026-05-17 / 2026-05-18) - context for the plan below

These landings shape the milestone scoping in the Updated milestone plan and
should be assumed available by every milestone:

- **9/9 native EBM primitives on Apple GPU.** `ebm_partition_exact` shipped
  as a single-dispatch stable-logsumexp MSL kernel; the Python-only EBM set
  is empty. Source: `_EBM_APPLE_GPU_FUSED` +
  `tessera.ebm.partition_exact_from_energies`.
- **Buffer-pool RAII hardening.** Every dispatcher in `apple_gpu_runtime.mm`
  acquires via `TS_METAL_BUF_ACQUIRE` / `TS_METAL_BUF_ACQUIRE_WITH_BYTES`
  macros that declare a stack-scoped `MetalBufferGuard`. Release runs on
  every exit path - success, early `return false;`, and exception - by
  construction. Locked by `tests/unit/test_apple_gpu_buffer_pool.py` (7
  regression tests including a registry/manifest status agreement check).
- **`@clifford_jit` AST -> CliffordIRProgram lowering at decoration time.**
  Replaced the older trace-capture path. Operand vocabulary covers
  function-arg Names, SSA `%tN` refs, and inline literal refs
  (`#int:N` / `#float:V` / `#bool:0|1`). The AST validation pass now
  requires the immediate receiver of every op call to be `ga` or an
  `Attribute` chain ending in `.ga.<op>` with a `Name` root - so
  `foo.norm(a)` and `np.linalg.norm(a)` are rejected. 6 negative tests
  lock the contract.
- **`primitive_coverage.py` promotion.** Added a `_partial` factory and
  promoted the 8 EBM ops with fused manifest entries from `planned` to
  `partial`. Stale `_planned` rows for natively-shipped primitives are
  caught by `test_native_ebm_ops_promoted_in_primitive_coverage`.

## Current source-base map

| Area | Primary source | Current role | Assessment |
|---|---|---|---|
| Public Python surface | `python/tessera/__init__.py`, op registry, `op_catalog.py` | User-facing ops, dtype and shape helpers, JIT entry points | Broad surface; good for demos, but coverage varies by layer. |
| General JIT frontend | `python/tessera/compiler/jit.py` | Decorator, source recovery, constraints/effects, Graph IR, compile bundle | Mature as a developer-facing artifact path; execution still routes through CPU/reference or target-specific special cases. |
| Textual frontend | `python/tessera/compiler/frontend/parser.py` | DSL parser to Graph IR, preserving control/schedule markers | Useful for spec conformance and examples; should become an end-to-end eval lane. |
| Graph IR | `graph_ir.py`, `TesseraOps.td` | Canonical object model plus MLIR interop text | Strongest semantic layer. Source spans and diagnostics are a real asset. |
| Schedule IR | `schedule_ir.py`, programming model sources | Mesh, pipeline, stage, layout, prefetch, artifact markers | Good verifier coverage; some advanced distributed/runtime semantics remain metadata. |
| Tile IR | `tile_ir.py`, tile/queue/attention C++ sources | Async movement, MMA, reductions, queue barriers, attention helpers | Good for artifact inspection and lit contracts; hardware execution remains backend-specific. |
| Target IR/codegen | `target_ir.py`, `src/compiler/codegen/*` | CPU, NVIDIA, ROCm, TPU, Apple, Metalium, Cerebras, Rubin CPX | Apple and CPU are most concrete; other targets should stay labeled artifact-only unless CI proves execution. |
| Runtime ABI | `src/runtime`, `python/tessera/runtime.py` | Host runtime, CPU backend, CUDA/HIP scaffolds, artifacts | CPU path is real; accelerator runtime needs staged hardware gates. |
| Primitive coverage audit | `primitive_coverage.py`, `standalone_primitive_coverage.md` | 12-axis contract status, category classifiers, lowering metadata, backend manifests | Already does much of the M0 job; needs one merged renderer and drift gate across compiler layers. |
| Clifford/GA | `python/tessera/ga`, `clifford_jit.py`, `src/solvers/clifford` | Structured algebra types, GA ops, dialect, Apple GPU kernels | Strong direction; constrained vertical slice exists, but it is separate from general JIT. Recent hardening requires GA calls to resolve through `ga.<op>` or a `.ga.<op>` chain. |
| EBM | `python/tessera/ebm`, `src/solvers/ebm`, Apple runtime | Energy primitives, samplers, partition/losses, fused Apple paths | Strong Python and Apple milestone: 9/9 native Apple GPU EBM primitives are promoted, including `ebm_partition_exact`; arbitrary energy function lowering remains a key gap. |
| Apple GPU runtime hygiene | `apple_gpu_runtime.mm`, `test_apple_gpu_buffer_pool.py` | RAII buffer-pool acquisition through `TS_METAL_BUF_ACQUIRE*` macros | Recently hardened: early-return paths release buffers by construction and are locked by regression tests. |
| Tests | `tests/unit`, `tests/tessera-ir`, benchmarks | Unit, lit, benchmark, doc drift checks | Broad and useful; needs a small number of canonical end-to-end evals. |

## Functional support by layer

### Frontends

The general `@tessera.jit` path:

- recovers source by `inspect`, explicit source, or source path;
- extracts constraints such as `Divisible`, `Range`, and `Equal`;
- infers effects and checks deterministic contracts;
- emits `GraphIRModule` and a compile bundle with trace events;
- can execute supported straight-line op graphs through the CPU/reference
  plan;
- emits Schedule, Tile, Target, and backend artifacts for supported target
  kinds.

The textual frontend:

- parses modules, functions, kernels, meshes, type aliases, constants, op
  calls, simple assignments, returns, and preserved control markers;
- lowers executable subsets into the same Graph IR object model;
- keeps source spans for diagnostics.

The Clifford frontend:

- `@clifford_jit(target="apple_gpu")` walks a constrained Clifford-only
  Python function AST into a `CliffordIRProgram`;
- validates every op against the manifest-backed Apple GPU fused set;
- dispatches through the JIT bridge and records route proof.

Gap: there are three frontends with three integration stories. They should not
be forced into one parser immediately, but they do need one shared compile
session model and one status taxonomy.

### Graph IR

Graph IR is the cleanest layer. It has structured objects, verification,
source-span diagnostics, canonical op-name mapping, numeric policy metadata,
KV cache objects, mesh/type/constant metadata, and textual MLIR output for
inspection.

The op catalog is very broad. That is useful for documentation and reference
dispatch, but it can obscure whether an op has a real lowering path. The
current CPU plan treats catalog membership as compilable and then falls back
to the Python op registry for many operations. That is reasonable for a
reference backend, but the compiler plan should report one consistent layered
status taxonomy:

- `api`;
- `frontend`;
- `graph_ir`;
- `schedule_ir`;
- `tile_ir`;
- `target_ir`;
- `runtime`;
- `bench`.

### Schedule IR

Schedule IR has real object-model verifiers for meshes, pipeline regions,
stages, memory spaces, prefetch, collectives, and debug artifacts. It is a
good staging layer for distributed and pipeline features.

Gaps:

- Multi-rank execution remains mostly mock/planner territory.
- Some semantics are metadata-only, especially debug artifacts and advanced
  control markers.
- Schedule decisions need to be traceable back to cost models and target
  capability records, not only static defaults.

### Tile IR

Tile IR captures execution and movement: `tile.async_copy`, `tile.wait_async`,
`tile.mma`, `tile.reduce`, queue ops, attention helper ops, and barrier/debug
markers.

Gaps:

- There is a split between Python object-model Tile IR and C++ MLIR dialect
  coverage. Both are useful, but they need explicit equivalence tests for
  canonical examples.
- The memory model still needs stronger verifier coverage for atomics,
  fences, happens-before, and failure diagnostics.
- Tile IR should expose target legality earlier, before Target IR generation.

### Target IR and codegen

The target matrix is broad:

- CPU and Apple CPU have executable paths.
- Apple GPU has executable paths for selected tensor ops and a notably strong
  GA/EBM path.
- NVIDIA, ROCm, TPU, Metalium, Cerebras, and Rubin CPX have meaningful
  artifact, dialect, inventory, or lit-testable coverage, but should not be
  described as production runtime without hardware-gated CI.

Gaps:

- Target capability records and backend manifests are not yet the sole source
  of truth for lowering choices.
- Native execution proof should be standardized as a first-class compile
  artifact field across targets, not only in Apple GA/EBM benchmark rows.
- The compiler should prefer explicit "unsupported on target" diagnostics to
  silent reference fallback when the user requested native execution.

## End-to-end integration assessment

The repo has pieces of several end-to-end routes:

| Route | Current status | What is proven | Main gap |
|---|---|---|---|
| Python `@jit` matmul/elementwise/attention to CPU | Implemented | Graph -> Schedule -> Tile -> Target artifacts, CPU/reference execution | Broader op claims need layer labels; performance path is proxy. |
| Textual DSL to Graph/Schedule/Tile contracts | Implemented subset | Parser preserves source shape and emits inspectable Graph IR | Needs canonical full-stack evals from `.tss` source to runtime/artifact. |
| MLIR lit pass pipelines | Lit-testable | C++ passes and dialect contracts for named phases | Need regular build/lit gate in documented validation spine. |
| Apple GPU tensor ops | Partial native | Selected MPS/MSL ops and fusions | Coverage matrix should distinguish single-op, fusion, dtype, and shape limits. |
| Clifford/GA Apple GPU | Strong vertical slice | 17/17 fused MSL primitives, manifest, bridge trace, benchmarks, `@clifford_jit` AST IR | Not yet unified with general Graph/Schedule/Tile/Target pipeline. |
| EBM Apple GPU | Strong but specialized | 9/9 native primitives including `inner_step`, `refinement`, `langevin_step`, `decode_init`, `bivector_langevin`, `sphere_langevin`, `self_verify`, `energy`, `partition_exact`; fused `ebt_tiny`; workload proof bits | Arbitrary `energy_fn` lowering and on-device RNG remain open. |
| NVIDIA/ROCm/TPU/Metalium | Artifact/lit/scaffold | Target IR contracts and inventories | Hardware execution and benchmark gates are future work. |

The next integration target should be a small set of canonical programs:

1. `matmul -> softmax -> matmul` attention block.
2. `conv2d -> norm -> activation` CNN block.
3. `kv_cache_append -> prune -> read` decode state block.
4. `rotor_sandwich -> norm` Clifford geometry block.
5. `decode_init -> inner loop -> self_verify` EBM block.
6. `rotor_sandwich -> ebt_tiny` GA+EBM block.

Each program should emit a single JSON compile report with:

- frontend source origin;
- Graph/Schedule/Tile/Target artifact hashes;
- target capability decisions;
- fallback/native reason;
- proof of execution when native;
- numeric correctness result when executable;
- performance row when benchmarked.

## Architectural gaps

### 1. Status language is not strict enough

The repo already uses terms like `implemented`, `lit-testable`,
`artifact_only`, `reference`, `fused`, and `planned`, and it already has
multiple partial schemas: `primitive_coverage.py` contract axes,
`BackendKernelEntry`, target capabilities, category classifiers such as
`_GRAPH_IR_LOWERING_BY_CATEGORY` and `_SHARDING_RULE_BY_CATEGORY`, and
metadata such as `graph_ir_lowering`.

The gap is not absence of schema. The gap is that these schemas overlap
without one renderer and one drift gate. This creates documentation drift and
makes it easy for broad primitive support to sound like native compiler
support.

Remedy: consolidate the existing schemas into a generated
`CompilerLayerStatus` report instead of adding another hand-maintained status
system.

Suggested axis values:

- `api`: absent, public, experimental;
- `frontend`: missing, parsed, source-span diagnostics, jit-lowered;
- `graph_ir`: missing, emitted, verified;
- `schedule_ir`: missing, emitted, verified;
- `tile_ir`: missing, emitted, verified;
- `target_ir`: missing, artifact, lit-tested;
- `runtime`: unsupported, reference, mock, native;
- `bench`: none, proxy, native-smoke, native-regression.

### 2. General JIT and Clifford JIT are separate lanes

`@clifford_jit` is valuable because it is honest and constrained, but the
compiler will eventually need Clifford values to appear in mixed tensor/GA/EBM
programs. Today that means bridging between the ordinary Graph IR path and a
separate `CliffordIRProgram`.

Remedy: following Decision #15a and `ga_scope_lock.md` Q2, keep
`Multivector` as a sibling value kind rather than a seventh tensor attribute.
If `EnergyState` and `ComplexField` are added, decide explicitly whether they
are sibling value kinds or annotations on an existing kind before
implementation. `@clifford_jit` can remain a fast-path frontend while sharing
the same compile session, artifact metadata, diagnostics, and target
capability checks.

### 3. Target capability should own lowering legality

Lowering code, backend manifests, and capability records overlap. The compiler
should not scatter "can this op run here?" decisions across multiple files.

Remedy: make capability lookup mandatory at every transition:

- Graph -> Schedule: is the op schedulable?
- Schedule -> Tile: is the memory/layout strategy legal?
- Tile -> Target: is the target instruction/kernel family available?
- Target -> Runtime: is execution native, reference, artifact-only, or
  unsupported?

### 4. Memory model is under-specified in enforcement

Async copies, queues, and barriers have good local coverage. Broader memory
model claims around atomics, fences, and happens-before need verifier and
negative tests.

Remedy: create a memory-model verifier milestone with negative lit fixtures
and Python object-model verifier parity.

### 5. Native runtime proof is uneven

Apple GA/EBM has good proof bits. Other accelerator paths are more often
artifact/lit-based.

Remedy: do not add a second proof struct. Extend `JitBridgeRoute`, which
already records `op_name`, `target`, `status`, `symbol`, `context`,
`latency_ms`, and argument summaries for GA/EBM proof bits. The route record
should become the common proof envelope by adding:

- hardware/device ID when available;
- correctness comparator;
- benchmark envelope.

### 6. EBM specialization is ahead of arbitrary function lowering

The native EBM path is strongest where kernels are specialized. General
energy functions still need lifting into target code, especially for true
per-step gradient recomputation.

Remedy: add a restricted energy-function compiler that reuses the
`clifford_jit` AST-to-IR template:

- single-return functions and straight-line assignments;
- whitelisted energy ops instead of whitelisted GA ops;
- SSA-style operand refs and inline literals such as `#int:N` and `#float:V`;
- AST subset for polynomial, bilinear, norm, activation, and small MLP heads;
- automatic `grad_y` generation;
- MSL first, then CUDA/HIP artifact lowering;
- explicit fallback diagnostics for unsupported Python.

The reusable infrastructure should be extracted before the EBM lowerer grows:
source recovery, AST validation, SSA naming, literal encoding, whitelist
resolution, and diagnostic formatting.

### 7. Sharding, numeric policy, and build gates need owners

Three important gaps are currently mentioned but not owned:

- Sharding/distributed execution remains a Phase G/H/I dependency. This plan
  should explicitly defer native distributed execution while still requiring
  planner/status reporting.
- `numeric_policy` already exists and is attached to many primitives, but
  GA/EBM dtype expansion must preserve `storage` versus `accum` distinctions,
  especially for fp16/bf16 kernels.
- Build/lit validation deserves its own milestone because C++/MLIR contracts
  are a separate failure mode from Python unit tests.

## Needham / Visual Complex Analysis opportunity

Needham's *Visual Complex Analysis* is useful for Tessera because it treats
complex analysis as geometry: local rotation plus scaling, conformality,
Mobius transformations, flow, potential, residues, and the Riemann sphere.
That lines up with the repo's new direction: make mathematical structure
visible to the compiler, then preserve or exploit it during lowering.

This should not become a vague "complex numbers are cool" feature. It should
be turned into compiler-visible invariants and primitives.

### Where it fits

| Needham idea | Compiler opportunity | Why it matters |
|---|---|---|
| Holomorphic maps are locally scale + rotation | `ComplexField` or 2D GA subalgebra with a `conformal` effect/constraint | The compiler can preserve angle structure and reject illegal transforms. |
| Cauchy-Riemann equations | Verifier pass for analytic kernels | Catch user-defined complex functions that break holomorphic assumptions. |
| Mobius transformations | Primitive family for fractional linear transforms and Riemann-sphere maps | Useful for geometry, graphics, conformal models, and stable coordinate changes. |
| Riemann sphere / stereographic projection | Manifold helper shared by GA and EBM | Complements current `Sphere` and future conformal geometric algebra work. |
| Harmonic functions and potential flow | Solver/EBM energy primitives based on Laplace structure | Lets EBM energies and PDE solvers share structure-aware passes. |
| Contour deformation and residues | Complex integration test or symbolic-numeric validation harness | Excellent for validating integration, branch cuts, and singularity diagnostics. |
| Visual branch cuts and singularities | Diagnostics for discontinuities, poles, and invalid domains | Helps compiler errors become mathematical rather than only type-level. |

### Proposed "Visual Complex" compiler lane

Do not start with a full complex-analysis system. Start with a small lane that
strengthens existing GA/EBM and solver work:

1. Add a `ComplexScalar` / `ComplexField` reference type or encode complex
   scalars as `Cl(0,1)` or as the even subalgebra of `Cl(2,0)`. `Cl(0,1)`
   is the direct two-component complex algebra; the `Cl(2,0)` even
   subalgebra is better when the feature wants to stay inside the existing
   Euclidean GA rotor story. The pilot must pin one choice in a decision log
   before implementation.
2. Add primitives: `complex_mul`, `complex_exp`, `mobius`, `stereographic`,
   `conformal_jacobian`, `laplacian_2d`.
3. Add a verifier for Cauchy-Riemann consistency on simple user functions.
   This requires the same restricted AST and symbolic/numeric derivative
   machinery as M6, so M7 should remain spec-only until the shared AST-to-IR
   path exists.
4. Add a Needham-style visual conformance suite:
   - grid under `z^2`;
   - grid under `exp(z)`;
   - Mobius map on circles/lines;
   - stereographic projection round trip;
   - contour integral around a simple pole.
5. Lower first to CPU/reference and Apple GPU artifact/native where trivial.
6. Connect to EBM by adding conformal energy examples on the plane/sphere.

The best immediate use is not raw performance. It is to make Tessera's
mathematical IR story sharper: the compiler can know when a transformation is
angle-preserving, when a scalar field is harmonic, and when an energy is being
optimized on a curved surface rather than in a flat array.

## Updated milestone plan

### M0 - Consolidated status taxonomy and source-of-truth cleanup

Goal: make support claims machine-checkable.

Deliverables:

- Consolidate existing status schemas rather than adding a parallel one:
  `primitive_coverage.py`, category classifiers, `metadata.graph_ir_lowering`,
  `capabilities.py`, `BackendKernelEntry`, and op catalog acceptance.
- Generate a compiler support table from `op_catalog.py`,
  `primitive_coverage.py`, `capabilities.py`, and `backend_manifest.py`.
- Pick the eight-layer taxonomy (`api`, `frontend`, `graph_ir`,
  `schedule_ir`, `tile_ir`, `target_ir`, `runtime`, `bench`) and remove
  duplicate vocabulary such as "graph-recognized vs reference-executable" from
  future docs unless it maps to those axes.
- Update docs that use broad terms like "implemented" to include layer labels.
- Add `claim_lint` as a doc-vs-manifest drift gate that fails when public docs
  claim native support without a runtime proof or benchmark row.

Acceptance:

- One generated table answers, for every important op family, which layers are
  API/front-end/Graph/Schedule/Tile/Target/runtime/bench covered.
- Existing docs link to the generated table rather than duplicating claims.

### M0.5 - Generated support-table CLI

Goal: make the M0 consolidation repeatable.

Deliverables:

- Add one CLI:
  `python -m tessera.compiler.audit support_table > docs/audit/generated/support_table.md`.
- Feed the CLI from the existing classifiers and manifests.
- Add a check mode that regenerates the table and compares it against the
  checked-in output.

Acceptance:

- CI fails when generated support status drifts from the checked-in table.
- The generated table includes enough provenance to show which source decided
  each axis.

### M1 - Canonical end-to-end compiler reports

Goal: make each frontend-to-codegen path inspectable in one format.

Deliverables:

- Add a `CompileReport` JSON schema that records source origin, IR hashes,
  target decisions, diagnostics, fallback/native status, and proof bits.
- Use extended `JitBridgeRoute` rows as the proof envelope for native
  execution rather than introducing a separate `NativeExecutionProof`.
- Emit reports from general `@jit`, textual frontend lowering, and
  `@clifford_jit`.
- Add the report schema plus two canonical programs first; the full six-program
  suite is M1.5.

Acceptance:

- CI can run the first canonical programs CPU-only.
- Apple GPU programs add native proof fields when available and skip cleanly
  when unavailable.
- Reports are deterministic except for timing fields.

### M1.5 - Canonical-program driver map

Goal: prevent the six-program suite from becoming a vague wish list.

Deliverables:

- Map each canonical program to an actual driver:
  - `matmul -> softmax -> matmul`: existing Apple fusion benchmark or a new
    compiler-report wrapper around it.
  - `conv2d -> norm -> activation`: new CPU/reference driver unless an
    existing example can be promoted.
  - `kv_cache_append -> prune -> read`: new compile-report driver.
  - `rotor_sandwich -> norm`: `benchmark_ga_ebm.py` / `@clifford_jit`.
  - `decode_init -> inner loop -> self_verify`: EBM benchmark/report driver.
  - `rotor_sandwich -> ebt_tiny`: existing GA+EBM workload report.
- Mark missing drivers explicitly rather than claiming the suite is complete.

Acceptance:

- Each canonical program has an owner file and a status row in the generated
  support table.
- At least two programs run in default CPU-only CI; Apple GPU variants skip
  cleanly when unavailable.

### M2 - Frontend unification without losing specialization  **(2026-05-18: landed — 5 of 5 sub-deliverables; both halves of Step 5 closed)**

**Status:** Steps 1–4 shipped + tested.  Step 5 **both halves
landed (2026-05-18):**

- **Python half** — canonical-IR equivalence harness at
  ``tests/unit/test_canonical_ir_equivalence.py`` + 5 golden files
  under ``tests/unit/canonical_golden_ir/``.  Byte-identity test
  parametrized over the 5 canonicals with deterministic
  ``_ir_text(...)``; catches silent Python-side IR drift.  13 tests.
- **C++ build half** — ``tessera-opt`` now builds against
  **MLIR 21** (Homebrew ``llvm@21``, version 21.1.8) via
  ``cmake --build build --target tessera-opt``.  Produces a
  69 MB binary registering 4 dialects (tessera / tessera.neighbors
  / tessera.solver / tessera_apple) + 70+ Tessera passes + 6
  named lowering pipelines.  Fixes that landed: (a)
  ``Tessera_DeltaAttentionOp`` and ``Tessera_LightningAttentionOp``
  gained the ``AttrSizedOperandSegments`` trait (MLIR 21 stricter
  check on multiple ``Optional<TensorType>`` operands); (b)
  ``HaloInferPass::getDeltaValues`` now handles
  ``DenseIntElementsAttr`` taps (the textual MLIR encoding for
  ``dense<[1,0]> : tensor<2xi64>``).  Smoke tested by
  ``tests/unit/test_tessera_opt_build.py`` (5 tests, skipped when
  binary absent).
- **Lit-side equivalence** — Phase 7 Neighbors lit fixtures (4 tests)
  all PASS via ``tessera-opt``; full ``tests/tessera-ir`` suite:
  34/72 PASS, 19 UNSUPPORTED (hardware-specific NVIDIA/ROCm), 19
  XFAIL (older), **0 FAIL**.

Landed (2026-05-18):

| Step | Capability | Module / Test |
|---|---|---|
| 1 | :class:`CompileSession` + ``compile_session()`` scope; capability cache + artifact-hash index + value-kind reduction | `python/tessera/compiler/compile_session.py` · `test_compile_session.py` (14) |
| 2 | ``tessera.bridge`` boundary ops (``multivector_to_tensor`` / ``tensor_to_multivector`` / ``complex_to_tensor`` / ``tensor_to_complex``); ``value_kind_of`` (function, not attribute — Decision #15a) | `python/tessera/bridge.py` · `test_bridge_boundary_ops.py` (15) |
| 3 | :class:`TesseraValueKindError` with source-span diagnostics; ``check_call_kinds`` helper; ``@clifford_jit`` runtime rejects non-Multivector arguments | (in `bridge.py`) · `test_mixed_op_diagnostics.py` (8) |
| 4 | Cross-frontend integration tests — three frontends in one session, schema parity, distinct ``value_kind`` per report, deduplication | `test_compile_session_integration.py` (8) |
| 5 | *Deferred:* Python Tile IR ↔ MLIR lit equivalence | Needs ``tessera-opt`` built; tracked as M2-Step-5 follow-up |

**Decision #15a structural lock:** ``value_kind_of`` is a
**function**, not a method/attribute on a sibling value (verified
by ``test_value_kind_of_is_a_function_not_a_method`` — a regression
that adds ``x.value_kind`` to any sibling type fails the test).
Per-report ``value_kind`` is preserved by the session (verified by
``test_session_does_not_mutate_per_report_value_kind``).

**Canonical-driver patch:** all 6 canonical drivers now route their
final ``CompileReport`` through ``finalize_compile_report`` so the
session sees every shipped program's report exactly once.
``rotor_sandwich_norm`` nests its inner ``@clifford_jit`` call in a
discarded capture scope to avoid double-emit (driver report
supersedes the decorator's auto-emit).

Step 5 follow-up scope (when ``tessera-opt`` build lands):

- Define canonical Tile IR text expectations for each of the 6
  canonical programs.
- Run ``tessera-opt`` to lower each program through the canonical
  pipeline and ``FileCheck`` against the expected output.
- Equivalence pass: Python object-model Tile IR must produce the
  same op sequence as the C++ MLIR lowering on the same input.

Goal: align general JIT, textual DSL, and Clifford JIT under one compile
session model.

Deliverables:

- Introduce a shared compile session object with diagnostics, target
  capability decisions, artifact hashes, and route trace storage.
- Let `@clifford_jit` produce a Graph-compatible module or attach its
  `CliffordIRProgram` as a typed submodule artifact.  **Per Decision
  #15a (locked in `ga_scope_lock.md` Q2), Multivector remains a
  sibling value kind, not a seventh tensor attribute** - M2 must keep
  that boundary explicit at the IR layer and in the compile-session
  schema (no "is this a Multivector?" check disguised as an attribute
  read).
- Add equivalence tests for Python object-model Tile IR versus C++ MLIR
  dialect/lit expectations on the canonical examples.
- Add mixed tensor + GA tests that prove values can cross the boundary through
  explicit ops, not hidden Python calls.

Acceptance:

- A GA demo and a tensor demo produce the same report envelope.
- Unsupported mixed operations fail with source-span diagnostics.
- The compile-session schema records value kind (`tensor` /
  `multivector` / ...) as a distinct field - not a dtype variant.

### M3 - Target legality and fallback hardening

Goal: prevent accidental over-claiming and silent native fallbacks.

Deliverables:

- Require target capability checks at every lowering transition.
- Add `native_required=True` or equivalent JIT option that errors instead of
  falling back.
- Make runtime fallback reasons stable and testable.
- Split CPU reference execution from CPU optimized execution in metadata.

Acceptance:

- Requesting an unsupported native target returns an actionable compiler
  diagnostic before runtime dispatch.
- Docs and tests can tell reference, artifact, and native paths apart.

### M4 - Memory model verifier

Goal: move async, queue, barrier, atomics, and happens-before claims from
documentation into verifier behavior.

Deliverables:

- Add Python object-model verifier checks for atomics/fences/happens-before
  metadata.
- Add negative MLIR lit fixtures for illegal async-copy, barrier, queue, and
  memory-space transitions.
- Connect diagnostics to source spans where possible.

Acceptance:

- Invalid memory programs fail before Target IR.
- The memory model spec points to concrete tests for every normative claim.

### M5 - Build, lit, and validation spine

Goal: make C++/MLIR contracts visible in the same validation spine as Python
unit tests.

Deliverables:

- Document which targets run Python unit tests, MLIR lit tests, C++ build
  checks, and hardware-gated runtime checks.
- Add check aliases or validation-script entries for the canonical lit suites.
- Standardize benchmark row fields across CPU, Apple, NVIDIA, ROCm, and TPU
  harnesses using the M1 route/proof envelope.
- Add hardware-gated smoke tests for accelerator paths without weakening
  CPU-only CI.

Acceptance:

- The validation guide distinguishes unit, lit, build-only, artifact-only, and
  hardware-runtime checks.
- Artifact-only paths cannot accidentally appear as native in generated
  reports.

### M6 - EBM arbitrary energy lowering

Goal: close the biggest semantic gap in the EBM compiler story.

Design precedent: the `@clifford_jit` AST -> `CliffordIRProgram` work
(2026-05-17/18) is the closest existing template - same shape (single
return, straight-line assignments, whitelisted callees, SSA `%tN`
operand refs, inline `#int:N` / `#float:V` literals).  M6 inherits
that shape; the difference is the whitelist (energy primitives, not
GA primitives) and the per-IR-op gradient pairing.

**Step 3 gate (2026-05-18 post-reassessment).**  Step 3 starts a
new IR layer (`grad_y` per op, fused energy+gradient kernels) and
deserves the same level of inspectability the rest of the compiler
just gained.  Hold Step 3 until **report/fallback proof is
uniformly visible**:

- [x] M0/M0.5 generated support table (drift-gated).
- [x] M0 `claim_lint` lifting public docs to manifest truth.
- [x] M1 `CompileReport` schema, two canonical drivers shipped.
- [x] M1.5: **all six canonical drivers shipped** —
  rotor_sandwich_norm, matmul_softmax_matmul,
  decode_init_inner_loop_self_verify, conv2d_norm_activation,
  kv_cache_append_prune_read, rotor_sandwich_ebt_tiny.
  (Honest-reporting note: matmul_softmax_matmul,
  kv_cache_append_prune_read, and conv2d_norm_activation
  emit ``fallback_reason=REFERENCE_FORCED`` even on Darwin
  because their driver bodies don't dispatch a fused kernel
  today — REFERENCE_FORCED is the M5/M3 contract for
  "manifest-capable but not yet natively executed".)
- [x] M3 `native_required=True` + stable `FallbackReason` enum.
- [x] **M3 follow-up — reference vs. optimized split in benchmark
  JSON (2026-05-18, closed).**  ``benchmarks/common/artifact_schema``
  ships ``ExecutionKind`` enum with four values
  (``reference``/``optimized_native``/``artifact_only``/``unknown``)
  attached to every ``BenchmarkRow`` and surfaced through
  ``to_dict()``/``flat_dict()``/``telemetry_for_row()``.  Independent
  of ``CompilerPath`` (how we got to runtime) and ``RuntimeStatus``
  (whether runtime executed).  Locked by
  ``tests/unit/test_benchmark_execution_kind.py`` (8 tests,
  CPU-only, non-slow).
- [x] M4 memory-model verifier (happens-before + memory-space).
- [x] M5 `BenchmarkRow` schema + validation spine doc + no-silent-native gate.
- [x] **M5 follow-up — `tessera-translate` (Python + C++) shipped
  (2026-05-18, closed; previously Python scaffold only).**
  Python CLI ``tessera-translate`` at
  ``python/tessera/cli/translate.py`` with **5 subcommands**:
  ``stablehlo``/``gguf``/``safetensors``/``info`` (wrapping
  ``tessera.aot`` exports) + ``mlir`` (pass-through to the C++
  binary).  ``pyproject.toml`` registers it as a console script.
  **C++ MLIR binary ``tessera-translate-mlir`` shipped
  (2026-05-18, closed):** new
  ``tools/tessera-translate/tessera-translate.cpp`` ships a thin
  ``mlirTranslateMain`` wrapper that registers every Tessera
  dialect (tessera / neighbors / TPP / Apple — conditional on
  the build feature flags) on top of **4 standard MLIR
  translations**: ``--mlir-to-llvmir`` / ``--import-llvm`` /
  ``--serialize-spirv`` / ``--deserialize-spirv``.
  ``tools/tessera-translate/CMakeLists.txt`` links
  ``MLIRTranslateLib`` + the LLVM-IR import/export libraries +
  the SPIR-V translate-registration libs + every Tessera
  dialect.  End-to-end smokes verified: (a) a tiny LLVM-dialect
  MLIR module translates to real LLVM IR text
  (``define i32 @add(...) { ... %3 = add i32 ... ret i32 %3 }``);
  (b) a ``spirv.module`` round-trips through ``--serialize-spirv``
  + ``--deserialize-spirv`` with SPIR-V magic ``0x07230203``
  preserved.  Locked by ``tests/unit/test_cli_translate.py``
  (6 tests) +
  ``tests/unit/test_tessera_opt_build.py::test_tessera_translate_mlir_*``
  (4 tests, including the documented-flag advert lock that
  prevents the README from drifting ahead of the binary's actual
  ``--help`` surface).
- [x] **M5 follow-up — TPP solver wired into ``tessera-opt``
  (2026-05-18, closed; previously Python frontmatter only).**
  ``python/tessera/solvers/tpp.py`` and
  ``python/tessera/solvers/__init__.py`` ship the
  ``tpp-space-time`` pipeline-alias name, 7 pass names, 2 type
  names, 2 attr names.  **C++ side now actively dispatches:**
  new ``src/solvers/tpp/include/tpp/InitTPP.h`` declares
  ``registerTPPDialect``/``registerTPPPasses``/``registerTPPPipelines``;
  ``src/solvers/tpp/lib/InitTPP.cpp`` registers all 7 passes via
  ``mlir::registerPass(...)`` plus the ``tpp-space-time``
  pipeline alias.  ``tessera-opt`` links the new
  ``TesseraTPPInit`` static library via the ``TESSERA_HAVE_TPP``
  compile-time guard.  Fixed an underlying dialect bug: TPP's
  ``TPP.td`` lacked ``useDefaultAttributePrinterParser = 1``, so
  attributes like ``#tpp.bc<"periodic">`` couldn't round-trip.
  ``status()`` now returns
  ``dialect_present=True / passes_present=True /
  pipeline_alias_present=True / lit_fixtures_runnable=True /
  python_driver_wired=False`` (honest about the remaining
  embedded-MLIR Python binding gap).  4/4 lit fixtures under
  ``src/solvers/tpp/test/TPP/`` pass.  Locked by
  ``tests/unit/test_solvers_tpp.py`` (7 tests) +
  ``tests/unit/test_tessera_opt_build.py::test_tpp_*`` (2 tests).
- [x] **`selective_ssm` Graph IR op (2026-05-18, closed).**
  Added ``tessera.selective_ssm`` op to
  ``python/tessera/compiler/op_catalog.py`` (state-space lowering
  kind, stateful effect, arity 4–6).  Flipped
  ``selective_ssm.graph_ir_lowering`` from ``missing`` to
  ``registered`` in ``primitive_coverage.py``.  Registry now
  reads **0 missing across all 374 entries** on the
  ``graph_ir_lowering`` axis (was 1).  ``PYTHON_API_SPEC.md`` row
  added so ``test_python_api_spec_lists_current_runtime_op_catalog``
  passes.  Dashboard snapshot in
  ``docs/audit/standalone_primitive_coverage.md`` regenerated.
- [x] M6 Step 1 + Step 2 — shared `ast_ir` core, `@energy_jit`
  whitelist + IR, lowering tests.
- [x] **CompileReport auto-emission (2026-05-18 reassessment Step 4)**:
  uniform `CompileReport` emission across `@tessera.jit`,
  textual, and `@clifford_jit` via `capture_compile_reports()`
  + `.compile_report()` accessors.  (Distinct from M6 Step 4
  Philox below — same number, different sprint.)
- [x] **Stability gate (closed 2026-05-18)**:
  `test_compile_report_stability_gate` parametrizes across
  every shipped canonical program (6 of 6) and asserts each
  emits a stable `report_hash()`, correct `fallback_reason`,
  and stable `frontend`/`value_kind`/`target` triple.
  **26 tests** pass; no hash collisions.  Includes the
  no-silent-native rule (target ≠ cpu + fallback_reason=None
  must carry a proof: route, ir_hash, or symbol).
- [x] **Step 3 start (2026-05-18)**: closed-form VJP table for the
  full 14-op energy whitelist landed in
  `python/tessera/compiler/energy_vjp.py`.  Every op has a
  finite-difference test (20/20 within 1e-3 to 1e-6 tolerance).
  This is the symbolic-grad core that M6 Step 4's MSL codegen
  will consume; M7's Cauchy-Riemann verifier can also reuse it.

- [x] **Step 3 grad_y + T-step refine (2026-05-18)**: shipped
  `python/tessera/compiler/energy_grad.py` with
  :class:`EnergyGradientProgram` (reverse-mode AD over the
  closed-form VJP table) and :func:`refine` (T-step gradient
  descent with build-once-reuse-per-step invariant — 16/16
  tests including a finite-difference check on `mlp_head`,
  convergence to a known quadratic minimum, and the
  build-call-count invariant verifying refine doesn't rebuild
  per step).
- [x] **Step 3 wired into a canonical (2026-05-18)**: the
  ``decode_init_inner_loop_self_verify`` canonical now exposes
  :func:`run_per_step_gradient` — a sibling entry point that
  uses :func:`refine` with per-step gradient recomputation.
  The shape matches what an MSL-fused energy+gradient kernel
  would emit.
- [x] **Step 4 — Philox-4x32-10 reference + MSL template
  (2026-05-18)**: `python/tessera/compiler/philox.py` ships
  the pure-Python Philox-4x32-10 (verified against three
  canonical reference vectors from the Random123 paper) plus
  :func:`philox_msl_source`, an MSL source template that
  matches the Python reference byte-for-byte (constants, round
  count, helper names).  18/18 tests including the cross-platform
  invariant lock.

- [x] **Step 4 runtime emission (2026-05-18)**: the Philox MSL
  template is now embedded in
  ``src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm``
  as the kernel ``ebm_langevin_step_philox_f32``.  New C ABI
  symbol ``tessera_apple_gpu_ebm_langevin_step_philox_f32(y,
  grad, eta, noise_scale, key[2], counter[4], out, n)``.
  Manifest entry ``ebm_langevin_step_philox`` resolves through
  ``jit_bridge.dispatch_via_manifest``.  Python wrapper
  ``tessera.ebm.langevin_step_philox`` falls back to a numpy
  reference (mirrors the MSL kernel byte-for-byte) when the
  Apple GPU runtime isn't available.  14/14 tests including
  the **cross-platform determinism check** that the Apple GPU
  output matches the Python reference within fp32 tolerance.

Remaining Step 4 work (deferred to a follow-up sprint, gated
on more apple_gpu_runtime.mm edits): the same on-device-RNG
pattern extended to ``decode_init`` and ``sphere_langevin``;
benchmark-row updates that surface the on-device path's
performance vs. the host-supplied-noise path.

Deliverables:

- **Step 1 - Lift the AST-to-IR core out of `clifford_jit`.**  Move
  `_ASTLowerer`, the constant-literal encoding, and the
  validate-against-manifest pass into a shared module (e.g.
  `tessera.compiler.ast_ir`).  `clifford_jit` keeps the GA
  whitelist; `energy_jit` adds its own whitelist.
- **Step 2 - Define `EnergyIRProgram`.**  Allowed callees: quadratic
  forms, bilinear forms, polynomial up to a fixed degree, vector
  norms, a small activation set (ReLU/Tanh/Sigmoid/GELU/Softplus),
  and small dense matmuls for MLP energy heads.
- **Step 3 - Generate `grad_y` per op.**  Each whitelisted op has a
  closed-form VJP w.r.t. the candidate variable.  Fuse energy +
  gradient kernels for T-step refinement so the inner loop never
  re-uploads gradients.
- **Step 4 - On-device RNG.**  Philox-4x32-10 in MSL so
  `langevin_step` / `decode_init` / `sphere_langevin` generate noise
  on-device from a 4-element key + counter.
- **Step 5 - CUDA/HIP artifact lowering for the same energy IR**
  (gated on Phase G/H).

Acceptance:

- A real EBT-tiny refinement recomputes gradient per step natively
  for the supported subset on Apple GPU.
- Unsupported energy functions fail with a precise "operation X not
  in the energy whitelist (line L, col C)" diagnostic - same shape
  as `clifford_jit`'s validation errors.
- The shared `ast_ir` module is reused by `clifford_jit` (regression:
  every existing clifford_jit test still passes against the extracted
  core).

### M7 - Visual Complex / conformal pilot  **(2026-05-18: landed — 7-step sequence below)**

**Status:** all 7 steps shipped + tested (87 new tests across 6
test files).  Pivoted from the plan's first-choice ``Cl(0,1)``
representation to a non-GA :class:`ComplexScalar` sibling kind
because Cl(0,1) and Cl(2,0) are not in the v1 GA allow-list
(``ga_scope_lock.md`` Q1).  The pivot matches Decision #15a:
``Multivector`` is a sibling kind, ``ComplexScalar`` is the
next sibling.

Landed (2026-05-18):

| Step | Capability | Module | Tests |
|---|---|---|---|
| 1 | ``tessera.complex`` namespace + :class:`ComplexScalar` sibling | `python/tessera/complex.py` | `test_complex_namespace.py` (10) |
| 2 | ``complex_mul``, ``complex_exp``, ``complex_conjugate``, ``complex_abs`` + conformality | — | `test_complex_mul_exp.py` (18) |
| 3 | ``mobius`` + circle-preservation + group law + pole policy | — | `test_complex_mobius.py` (7) |
| 4 | ``stereographic`` / ``stereographic_inverse`` (S² ↔ ℂ) | — | `test_complex_stereographic.py` (12) |
| 5 | ``conformal_jacobian``, ``laplacian_2d`` (5-point stencil) | — | `test_complex_jacobian_laplacian.py` (12) |
| 6 | ``check_cauchy_riemann`` (numerical) + ``@analytic`` + ``NotHolomorphicError`` | — | `test_complex_cauchy_riemann.py` (21) |
| 7 | ``conformal_energy_on_sphere`` → ``tessera.energy.norm_sq`` composition | — | `test_complex_conformal_energy.py` (7) |

Needham-completion bundles (2026-05-18, landing on top of M7):

- [x] **Bundle B — cross-ratio + Möbius constructions (Needham Ch. 3)**:
  `cross_ratio(z1, z2, z3, z4)` (the fundamental Möbius
  invariant), `is_concyclic(...)` (real-valued cross-ratio test
  for four points on a generalized circle), and
  `mobius_from_three_points(src, dst)` (the canonical via-(0, 1, ∞)
  construction).  19 tests including Möbius-invariance of the
  cross-ratio, line-as-generalized-circle handling, ∞-point
  triples in the Möbius constructor, round-trip identity, and
  cross-ratio preservation by the constructed Möbius.
- [x] **Bundle A — log / arg / pow + Wirtinger (Needham Ch. 2, 4-5)**:
  `complex_arg` and `complex_log` (principal branch, NumPy-compatible
  cut along the negative real axis — `log(-1) = +iπ` locked by
  test); `complex_pow(z, w) = exp(w · log(z))`; **Wirtinger
  primitives `dz` and `dbar`** as first-class operators (where
  `dbar(f, z₀) = 0` is exactly the Cauchy-Riemann condition).
  37 tests including branch-cut policy, log↔exp round-trip,
  agreement with `numpy.log`/`numpy.power`, Wirtinger identities
  on `z²`, `e^z`, `z̄`, `|z|²`, and a lock that the existing
  `check_cauchy_riemann` residual equals `|dbar(f, z₀)|`.
- [x] **Bundle C — integration theory (Needham Ch. 7-9)**:
  `python/tessera/contour.py` ships `Contour` + `circle` /
  `line_segment` / `polygon` constructors, `contour_integral`
  (per-segment composite Simpson's, ε-nudged boundary samples
  for polygons with discontinuous γ'), `winding_number`
  (argument-variation method), `residue` (Cauchy formula), and
  the worked-example helpers `argument_principle_count` +
  `residue_theorem_sum`.  28 tests including Cauchy's theorem
  (∮ analytic dz = 0), Cauchy's integral formula
  (∮ dz/(z−a) = 2πi inside, 0 outside), winding numbers for
  CCW/CW circles, the argument principle counting zeros minus
  poles, and the residue theorem closing identity
  (∮ f dz = 2πi · Σ residues).
- [x] **Lower-priority Needham additions (lifted from
  research follow-ups, 2026-05-18)**: shipped all six items
  that were originally filed for "later if a workload motivates
  it":
  - **`complex_sqrt(z, branch=...)`** with explicit branch
    selection (Ch. 2, 8); zero special-cased; matches
    `numpy.sqrt` on the principal branch.  8 tests.
  - **Hyperbolic primitives** (`tessera.hyperbolic`, Ch. 6):
    `poincare_distance`, `upper_half_plane_distance`, Blaschke
    disk-automorphism isometry helper, Cayley transform
    `H⁺ ↔ 𝔻`.  Locks triangle inequality, isometry, and
    cross-model consistency.  12 tests.
  - **`flow_lines(f, ...)`** (`tessera.flow`, Ch. 10): 4th-order
    Runge-Kutta streamline tracer for `dz/dt = f(z)` with
    escape-radius guard; verified on constant / radial /
    rotational / diagonal flows.  7 tests.
  - **Riemann surface lifting** (`tessera.riemann_surface`,
    Ch. 12): `RiemannSurfacePoint(z, branch)`, `lift_sqrt`
    (2 sheets), `lift_log` (∞ sheets), and
    `follow_path_on_riemann_surface` — walking CCW once around
    0 flips the sqrt sheet (the canonical demo) and advances
    log's branch by +1.  10 tests.
  - **Schwarz–Christoffel mapping** (`tessera.conformal_advanced`,
    Ch. 12): prevertex-given MVP via direct numerical integration
    of the SC integrand.  4 tests.  **Follow-up
    `schwarz_christoffel_parameter_solve` (2026-05-18) closed:**
    full parameter-problem solver — given polygon vertices, solve
    for the prevertices via damped Newton on side-length-ratio
    residuals.  Fixes three prevertices by Möbius gauge
    (``x_1 = -1``, ``x_2 = 0``, ``x_N = +∞``) and parameterizes
    the remaining ``N−3`` free prevertices via log-spacings so
    the strict ordering ``0 < x_3 < x_4 < ... < x_{N-1}`` is
    structural rather than constrained.  Finite-difference
    Jacobian; line-search step damping; integrable endpoint
    singularities handled via ε-inset Simpson + ``t = x_lo + 1/u``
    substitution for the segment to ``+∞``.  6 tests (triangle
    short-circuit, vertex-count guard, clockwise-polygon
    rejection, unit-square convergence, regular-pentagon
    ordered prevertices, initial-guess length check).
  - **Weierstrass ℘ elliptic function** (`tessera.conformal_advanced`,
    Ch. 5): truncated-lattice-sum `weierstrass_p`,
    `weierstrass_p_derivative`, `weierstrass_invariants`.
    Verified evenness, two-period periodicity (1e-2 tolerance
    for cutoff=16), and the elliptic-curve identity
    ``℘'² = 4℘³ − g₂℘ − g₃``.  7 tests.  **Follow-up
    `weierstrass_p_adaptive` (2026-05-18) closed:** adaptive
    lattice-sum that grows the radius ring-by-ring until the
    latest ring's contribution falls below ``tol``.  Returns
    ``(value, cutoff_used, last_ring_magnitude)`` so callers
    see both the converged value and the truncation error
    bound.  6 tests (origin rejection, agreement with fixed
    cutoff, error-bound monotonicity, tolerance-relaxes-cutoff
    contract, evenness preserved, three-tuple shape).

Follow-ups (now closed — 2026-05-18):

- [x] **Symbolic Cauchy-Riemann path**: shipped
  `python/tessera/compiler/complex_jit.py` — re-uses the
  shared ``ast_ir`` core for lowering, ships a dedicated
  complex/conformal whitelist with explicit
  `HOLOMORPHIC_OPS` vs `NON_HOLOMORPHIC_OPS` classification,
  and exposes ``analytic_symbolic`` as a compile-time
  decorator that raises :class:`NotHolomorphicError` at
  decoration (no probing).  Walks the IR and names the
  offending op + python attr.  22 tests:
  whitelist-partition contract, IR lowering, holomorphic
  positives (``z²``, ``e^z``, ``mobius``, chained
  compositions), non-holomorphic negatives (``z̄``, ``|z|``,
  ``z·|z|``), agreement with the numerical CR verifier on
  shared cases, and the structural reuse lock (the M6
  ``ast_ir`` core is the single AST→IR backbone).
- [x] **MSL codegen for the conformal-primitive surface**:
  4 fused MSL kernels shipped in
  `src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm`
  for the GPU-beneficial ops — ``complex_mul``,
  ``complex_exp``, ``complex_stereographic``,
  ``complex_mobius``.  Each has an RAII-pool dispatcher,
  an extern "C" wrapper, a host-side reference fallback,
  and a new ``_COMPLEX_APPLE_GPU_FUSED`` manifest table
  routed via ``complex_manifest_for`` and
  ``lookup_apple_gpu_symbol``.  Python wrappers in
  ``tessera.complex`` dispatch f32 same-shape inputs
  through ``jit_bridge`` and fall back to the existing
  numpy paths otherwise.  23 tests including Apple-GPU
  cross-platform determinism for each of the 4 kernels.

  **Host-only by design** (no manifest entry):
  ``complex_conjugate`` (one float negation),
  ``complex_abs`` (sqrt of two squares),
  ``conformal_jacobian`` (4-call host composition),
  ``laplacian_2d`` (small stencil where host numpy beats
  a GPU dispatch round-trip at our typical sizes).  The
  manifest documents this honestly — a future workload
  that motivates GPU paths for these can re-promote.

Goal: use Needham's visual/conformal ideas to extend the mathematical IR lane.

Dependency: the Cauchy-Riemann verifier needs symbolic partial
derivatives on a Python AST subset - the same machinery M6 builds
for `grad_y` over the energy-op whitelist.  **M7 should land after
M6 so the symbolic-grad core can be reused.**  Keep M7 spec-only
until M6's `ast_ir` module exists.

Deliverables:

- **Pin the complex-scalar representation first.**  Three candidates:
  - `Cl(0,1)` - a single basis vector with `e1^2 = -1`.  The
    canonical 2-D complex algebra and the simplest path.
  - Even subalgebra of `Cl(2,0)` - isomorphic to C but a derived
    path; useful only if we also want to interoperate with 2-D GA
    rotations in the same program.
  - `ComplexScalar` sibling kind - non-GA, parallels Decision #15a's
    Multivector treatment.  Best if complex doesn't share enough
    machinery with GA to justify reuse.
  Recommendation: start with **`Cl(0,1)`** (smallest surface area;
  reuses the existing `ga.Cl(p, q, r)` constructor; defers the
  sibling-kind question until concrete demand appears).
- Add a small complex/conformal reference namespace
  (`tessera.complex` or `tessera.conformal`).
- Add `complex_mul`, `complex_exp`, `mobius`, `stereographic`,
  `conformal_jacobian`, `laplacian_2d` primitives.  CPU/reference
  first; Apple GPU artifact-only.
- **Cauchy-Riemann verifier.**  Reuses M6's symbolic-grad core on a
  user-supplied complex function: compute `du/dx`, `du/dy`, `dv/dx`,
  `dv/dy` and check `du/dx = dv/dy` and `du/dy = -dv/dx` symbolically.
  Diagnostic fires when a function is `@analytic`-marked but the
  identities fail.
- Add visual/numeric conformance fixtures for conformality and contour
  integration.
- Connect one conformal energy example to EBM (e.g., a harmonic
  potential on the plane lifted via stereographic projection).

Acceptance:

- CPU reference tests prove conformality for `z^2`, `exp(z)`, and Mobius
  examples.
- A diagnostic catches a non-holomorphic function when marked analytic.
- One EBM example runs over plane/sphere coordinates using the new primitives.
- The Cauchy-Riemann verifier shares its symbolic-grad core with M6
  (regression: M6's `grad_y` tests still pass after the extraction).

### M8 - Accelerator backend expansion

Goal: extend beyond Apple only after the proof format and legality gates exist.

Deliverables:

- NVIDIA: pick one GA primitive and one EBM primitive for CUDA artifact plus
  hardware smoke.
- ROCm: pick one MFMA-compatible tensor path plus one solver/EBM candidate.
- TPU/Metalium/Cerebras: keep artifact-only unless a real runtime proof lands.
- Numeric policy: extend `storage`/`accum` metadata to GA/EBM dtype expansion
  before promoting fp16/bf16 GA kernels.
- Sharding/distributed: defer native distributed execution to Phase G/H/I, but
  require planner and generated-status rows to stay current.

Acceptance:

- Each promoted backend has at least one native proof row and one negative
  unsupported-op diagnostic.

## Recommended next 30 days

1. Weeks 1-3: land M0 and M0.5 as consolidation of existing schemas, with the
   generated support table and drift gate.
2. Weeks 3-4: land the M1 report schema plus two canonical programs, not the
   full six-program suite.
3. Week 4: add the M3 `native_required=True` option and stable fallback
   diagnostics.
4. M6: write the design only, explicitly extracting the `clifford_jit`
   AST-to-IR template; no energy lowering code until M0/M1 are in place.
5. M7: keep the Visual Complex lane spec-only and pin the `Cl(0,1)` versus
   `Cl(2,0)` decision before any implementation.

Each milestone that lands should append or update an Architecture Decision row
in `CLAUDE.md` so the plan becomes part of the project's durable decision log
rather than a standalone audit proposal.

## Bottom line

The compiler is in a good but delicate phase. It has enough frontends, IR
layers, target artifacts, runtime paths, and mathematical primitives to be
interesting. The next improvements should make that power easier to trust:
precise support labels, reproducible end-to-end reports, fewer split-brain
frontend paths, stricter target legality, and a small number of native paths
that prove the whole compiler rather than only individual pieces.

The Needham-inspired direction is worth pursuing because it strengthens the
same thesis as the Clifford/EBM work: mathematical structure should be in the
IR, not hidden in tensor conventions. Start with conformality, Mobius maps,
Riemann-sphere projection, and Cauchy-Riemann diagnostics; let performance
follow once the semantics are crisp.
