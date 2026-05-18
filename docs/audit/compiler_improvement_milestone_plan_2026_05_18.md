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

### M2 - Frontend unification without losing specialization

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
- [x] M1.5: four canonical drivers shipped (rotor_sandwich_norm,
  matmul_softmax_matmul, decode_init_inner_loop_self_verify,
  conv2d_norm_activation).
- [x] M3 `native_required=True` + stable `FallbackReason` enum.
- [x] M4 memory-model verifier (happens-before + memory-space).
- [x] M5 `BenchmarkRow` schema + validation spine doc + no-silent-native gate.
- [x] M6 Step 1 + Step 2 — shared `ast_ir` core, `@energy_jit`
  whitelist + IR, lowering tests.
- [x] **Step 4 (2026-05-18 reassessment)**: uniform `CompileReport`
  emission across `@tessera.jit`, textual, and `@clifford_jit`
  via `capture_compile_reports()` + `.compile_report()` accessors.
- [x] **Gate (closed 2026-05-18)**: `test_compile_report_stability_gate`
  parametrizes across every shipped canonical program (6 of 6) and
  asserts each emits a stable `report_hash()`, correct
  `fallback_reason`, and stable `frontend`/`value_kind`/`target`
  triple.  40 tests pass; no hash collisions.
- [x] **Step 3 start (2026-05-18)**: closed-form VJP table for the
  full 14-op energy whitelist landed in
  `python/tessera/compiler/energy_vjp.py`.  Every op has a
  finite-difference test (20/20 within 1e-3 to 1e-6 tolerance).
  This is the symbolic-grad core that M6 Step 4's MSL codegen
  will consume; M7's Cauchy-Riemann verifier can also reuse it.

Step 3 remaining work: per-IR-op `grad_y` materialization at the
codegen layer, fused energy+gradient kernels for T-step refinement,
and on-device Philox RNG in MSL.  Steps 3+4 ride on top of the
audit + report machinery rather than landing parallel proof paths.

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

### M7 - Visual Complex / conformal pilot

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
