# Tessera Compiler Test Plan

Status: active test plan

Owning architecture plan:
[`INTEGRATED_COMPILER_PLAN.md`](../docs/audit/compiler/INTEGRATED_COMPILER_PLAN.md)
§0, synchronization key `E2E-REAL`. This document owns **how compiler claims
are tested**. Backend audit plans own the exact-device evidence and promotion
decision for their architectures.

This plan covers the current compiler architecture:

```text
Python @jit frontend
  -> Graph IR
  -> Schedule IR
  -> Tile IR
  -> Target IR
  -> backend IR / native image
  -> launch descriptor
  -> runtime execution
```

Merely having a non-empty artifact at every level is not an end-to-end proof.
The compiler-boundary claim requires an unbroken parent/child lineage:

```text
next.input_digest == previous.output_digest
```

Every stage must also record its producer, representation, target, and contract
version. A package synthesized again from the original Graph module is a useful
backend package, but it is **forked**, not compiler-boundary E2E. Tests must use
the terms `stage_complete`, `package_launchable`, and `lineage_complete`
separately.

## Test Layout

| Suite | Directory | Default Run | Purpose |
| --- | --- | --- | --- |
| Unit | `tests/unit/` | `pytest` and `scripts/test.sh` | Fast correctness contracts for Python compiler APIs, IR emission, pass preconditions, CPU proxy execution, and compiler documentation drift. |
| Performance | `tests/performance/` | `cmake --build . --target check-tessera-performance` or `TESSERA_RUN_PERFORMANCE_TESTS=1 ./scripts/test.sh` | Deterministic roofline/proxy performance contracts for compile latency, generated-artifact size, GEMM, attention, collectives, and benchmark report schema. |
| MLIR lit | `tests/tessera-ir/` | `check-tessera-ir` | FileCheck-based C++/MLIR pass and pipeline contracts. |
| Exact-device | `tests/device/` plus backend-owned unit/device suites | Explicit per-architecture release commands | Reference-vs-compiled execution on Apple, NVIDIA, ROCm, x86/AVX-512, and access-gated AMX hardware. |
| Numerical validation | `tests/tessera_numerical_validation/` | opt-in pytest suite | Cross-backend reference comparisons, dtype tolerances, determinism, and numerical-policy checks. |
| Project Evals | `tests/unit/`, `docs/context/`, generated context outputs, and future `tests/evals/` | opt-in locally, scheduled in CI | Whole-project coherence checks for specs, docs, samples, compiler pipelines, diagnostics, compatibility, context graph freshness, and health signals. |

## Validation Tier Matrix

| Tier | Scope | Default |
| --- | --- | --- |
| Tier 0 contract | Registries, documentation drift, artifact schemas, lineage state, diagnostic codes, and deterministic serializers | Every PR |
| Tier 1 Python unit | Frontend, Graph/Schedule/Tile object models, canonical orchestration, package selection, and runtime rejection paths | Every PR |
| Tier 2 MLIR/compiler | Positive and negative lit coverage for dialects, verifiers, each real boundary pass, and host-free target lowering | Every affected compiler PR when `tessera-opt` is built |
| Tier 3 package/runtime | Native-image, descriptor, cache, ABI-binding, fallback-disabled launch, and process cleanup tests; hardware-free where possible | Every affected package/runtime PR |
| Tier 4 exact-device correctness | Apple7, SM120, gfx1151, Zen 5 AVX-512, and separately access-gated AMX numerical execution | Required before an architecture execution claim lands |
| Tier 5 performance/promotion | Architecture-owned baseline comparison with valid timing domain and resource evidence | Required before selector/default promotion |
| Tier 6 breadth/fleet | Multi-op, dynamic, stateful, multi-rank, second-device, and long-running stability packets | Scheduled or release gate |

Hardware-free CI may prove parsing, verification, lowering, packaging schema,
and rejection behavior. It may not prove execution placement, numerical
correctness on an accelerator, or performance. Conversely, a direct backend
kernel test does not prove that the frontend and all intermediate IR boundaries
were traversed.

## E2E-REAL Implementation Matrix

Rows marked **planned** become required in the PR that implements the owning
work item. Until then, meta-tests require the row and its failure gate to remain
in this plan; they do not fabricate a passing implementation result.

| Work item | State at 2026-08-04 | Test deliverables | Required failure proof |
| --- | --- | --- | --- |
| E2E-REAL-0 — lineage truth | planned | Add `tests/unit/test_compiler_artifact_lineage.py`; extend orchestration, native-artifact, AOT/cache, and runtime serialization tests. | Substituted Tile text, wrong parent digest, missing producer/version, target mismatch, and cache reload with broken lineage must produce `lineage_complete = false` or fail construction. Stage presence must remain independently observable. |
| E2E-REAL-1 — real Schedule IR | planned | Add Schedule dialect parse/print/verify lit fixtures; build/link/registration checks; production `tessera-opt --show-dialects` and `--help` assertions; retain Python Schedule tests as oracle coverage. | Unknown/missing schedule contract, malformed content hash, duplicate pass registration, and metadata-only claims of value lowering fail. Fixtures must not use `--allow-unregistered-dialect`. |
| E2E-REAL-2 — scheduled matmul to launch Tile | planned | Add `tests/tessera-ir/phase2/e2e_matmul_graph_schedule_tile.mlir` plus invalid companion for x86-f32 and ROCm-f16/f32; compare selected Python-oracle schedule fields with the C++ result. | The Graph matmul or schedule decision surviving after conversion, missing six-operand launch ABI, lost storage/accumulation/layout/schedule metadata, invalid shapes, unsupported dtypes, and unsupported dynamic forms fail closed. |
| E2E-REAL-3 — x86 and ROCm consumers | planned; backend components individually proven | Add one host-free package-input test per backend and exact-device tests for Zen 5 AVX-512 and gfx1151. Both use one semantic case ID with target-specific typed Graph instances and independently complete digest chains. | Calling canonical compiler packaging with the original Graph module after Schedule/Tile exists, reference fallback, wrong target, stale image, missing descriptor, and ABI mismatch fail before launch. |
| E2E-REAL-4 — promotion | planned | Extend the existing gfx1151 pipeline-vs-direct benchmark and the x86 AVX-512 ratchet with lineage and artifact identities. | Invalid/zero timing, mixed timing domains, numerical failure, missing resource record, or regression against either the committed floor or same-run production lane rejects promotion without weakening correctness status. |
| E2E-REAL-5 — family breadth | planned | Parameterize the lineage harness over softmax/reduction, attention forward/backward, then stateful/training families; add NVIDIA SM120 and Apple-host packets when migrated. | A family cannot inherit another family's package, ABI, numerical tolerance, device evidence, or promotion decision. |
| E2E-REAL-6 — authority deletion | planned | Extend Decision #31 governance inventory to classify every second implementation as `production`, `candidate`, or `oracle`; add deleted-entry negative import/registration tests. | More than one production implementation for a boundary, an unclassified resynthesizer, or an oracle without a differential test fails CI. |

### Artifact Lineage Contract

| Artifact | Required input identity | Required output evidence |
| --- | --- | --- |
| Graph | source/trace digest and frontend producer | verified Graph text digest plus source provenance |
| Schedule | Graph output digest | verified mixed-level program digest, selected decisions, and named drops |
| Tile | Schedule output digest | verified logical/launch Tile digest and preserved numeric/layout/memory contracts |
| Target | Tile output digest | verified architecture Target IR digest and target identity |
| Backend IR | Target output digest | LLVM/ROCDL/NVVM/MSL/C-source digest and toolchain identity |
| Native image | Backend or Target output digest, as declared by the producer | payload digest, format, architecture, entry points, compile state, and device-library identities where applicable |
| Launch descriptor | Native-image digest | ABI, buffers/scalars, shape guards, launch geometry, ordering, dynamic memory, and provenance |
| Runtime result | descriptor digest plus bound argument contract | placement, execution kind, no-fallback state, numerical result, and timing provenance when measured |

No test may infer adjacency by comparing op names or by checking that both
artifacts are non-empty. Hash equality alone is also insufficient unless the
producer and contract version identify what the hash represents.

### First Vertical-Slice Corpus

The first shared workload is static rank-2 matmul. It is one semantic fixture
with target-specific typed instances and physical evidence, not two separately
authored backend programs. The initial x86 instance is f32. The initial ROCm
instance is f16 storage with f32 accumulation/output because that is the
existing gfx1151 WMMA and 8.02 TFLOP/s evidence contract. Their case ID and
semantic shape class match; their target-specific Graph digests do not.

| Class | Required cases | Purpose |
| --- | --- | --- |
| Structural smoke | one small aligned static shape for each bounded dtype contract | Fast Graph→Schedule→Tile parse/verify/FileCheck and digest adjacency |
| Aligned device | x86 f32 plus the existing ROCm f16/f32 WMMA-aligned case | ABI, grid, accumulator, dtype policy, and descriptor execution |
| Ragged device | gfx1151 f16/f32 `(33, 17, 31)` plus an x86 f32 non-vector-width-aligned case | Bounds, tails, K remainder, and no hidden shape specialization |
| Performance | gfx1151 `(2048, 2048, 2048)` for the committed 8.02 TFLOP/s comparison; x86's committed AVX-512 ratchet shape(s) | Architecture-specific promotion only; correctness cannot be inferred from timing |

Every numerical run uses deterministic inputs and compares both to the shared
reference and, while migration is active, to the retained production lane.
Bit identity is required only where the numeric policy and existing backend
contract promise it; otherwise the declared dtype tolerance is the gate.

## Change-to-Gate Routing

| Change surface | Minimum required gates |
| --- | --- |
| Artifact/lineage schema | Tier 0, Tier 1, cache/AOT serialization, old-artifact compatibility or explicit version rejection |
| Schedule/Tile ODS or boundary pass | Tier 0, Tier 1, all affected Tier 2 positive/negative fixtures, metadata-obligation verifier, backend-plan assessment |
| Shared package/descriptor/runtime ABI | Tier 0-3 for every registered target; exact-device rerun for each changed physical consumer |
| x86 lowering or generated code | Host-free x86 fixtures plus Zen 5 AVX-512 Tier 4; AMX evidence remains separate and access-gated |
| ROCm lowering or generated code | Required host-free ROCm compiler lane plus gfx1151 Tier 4; check intended ROCm toolchain/device visibility first |
| NVIDIA lowering or generated code | Host-free NVIDIA compiler fixtures plus SM120 Tier 4 on its owning host |
| Apple lowering or generated code | Host-free structural tests where applicable plus Apple7 Tier 4 on the Mac |
| Schedule choice, selector, or default | All correctness gates plus Tier 5 on the owning architecture; sibling evidence never transfers |
| Documentation-only plan change | Audit/docs/context-generation gates; no exact-device claim may change without its evidence packet |

All project tests run in the host environment. CUDA and ROCm commands must
first verify the intended device and toolchain are visible. A skip caused by
missing hardware is an honest access result, not validation of the feature.

## Project-Level Eval Matrix

Project evals sit above unit tests and answer whether Tessera remains coherent
as a product, compiler stack, and contributor surface. They should reuse the
fastest existing test mechanism for each concern until a dedicated
`tests/evals/` harness exists.

| Eval Family | Required Coverage | Gate |
| --- | --- | --- |
| Spec conformance | Canonical API symbols, named pipeline aliases, dialect symbols, and README phase/status claims match the implementation. | No stale public contracts or unsupported status claims. |
| End-to-end compiler evals | Representative Python/API samples produce lineage-linked Graph, Schedule, Tile, Target, backend, image, descriptor, and runtime evidence; package-only and compiler-boundary E2E remain distinct. | Every adjacent digest joins, the declared producer consumed the preceding artifact, semantic metadata survives or has a named drop, and fallback is disabled for execution claims. |
| Numerical correctness | Supported operators compare against NumPy/PyTorch-style references across shape classes, seeds, and dtype-specific tolerances. | Results stay within the declared tolerance for each dtype and backend mode. |
| Shape/layout evals | Symbolic shapes, tile boundaries, layout transforms, sharding plans, halo inference, and neighborhood topology cases. | Valid programs infer stable metadata; invalid programs fail before lowering. |
| Diagnostics quality | Invalid programs for effects, distributions, target support, memory spaces, and shapes produce stable useful errors. | Diagnostics include source context, compiler stage, violated invariant, and actionable category. |
| Documentation evals | Documentation code blocks, links, referenced paths, public symbols, and pipeline diagrams stay current. | Executable examples run or are explicitly marked pseudo-code; all referenced project paths exist. |
| Sample/tutorial evals | Getting-started samples and tutorials import, use canonical APIs, and produce expected outputs or artifacts. | CPU-promised samples run without accelerators and finish within a local smoke-test budget. |
| Agent context graph evals | Ontology, knowledge map, generated JSON/Markdown outputs, and agent workflows stay parseable, path-valid, deterministic, and marked non-authoritative. | `scripts/generate_context_outputs.py --check` passes and generated outputs match source YAML plus this test plan. |
| Compatibility/project health | CPU-only build path, optional hardware gates, CLI help, import-time budget, package metadata, and script/README command agreement. | Deterministic health checks pass without requiring hidden local state. |

### Eval Tiering

| Tier | Eval Scope | Default |
| --- | --- | --- |
| Fast local | Spec conformance, documentation smoke checks, context graph generation checks, sample import checks, and CLI/package health. | Developer opt-in and cheap enough for pre-commit use. |
| CI deterministic | Project evals that require no accelerator, including context graph output freshness, and do not depend on machine-specific timings. | Always on once the corresponding eval harness exists. |
| Scheduled | Numerical sweeps, broader sample execution, documentation execution, and performance regression checks. | Nightly or weekly depending on runtime cost. |
| Hardware-marked | Apple7, NVIDIA SM120, ROCm gfx1151, x86/AVX-512, access-gated AMX, and distributed backend evals. | Opt-in through the architecture-owned exact-device command; required before changing that architecture's execution or promotion claim. |

## Unit Test Matrix

| Compiler Area | Required Unit Coverage | Representative Tests |
| --- | --- | --- |
| Frontend source recovery | inspect source, explicit `source=`, unavailable source diagnostics | `test_end_to_end_matmul_cpu_path.py` |
| Constraint extraction | `require(...)`, invalid bindings, symbolic skip | `test_constraints.py` |
| Effect inference | pure/random/state/collective/determinism contracts | `test_effects.py`, `test_deep_learning_semantic_core.py` |
| Graph IR emission | function args, Region effects, nested op extraction, keyword attrs | `test_graph_ir.py`, `test_lowering_chain.py` |
| Artifact lineage | parent/child digests, producer/version identity, fork detection, serialization, cache/AOT joins | planned `test_compiler_artifact_lineage.py`, `test_e2e_spine_orchestration.py`, `test_native_artifact_contract.py` |
| Schedule IR boundary | registered dialect, mixed-level SSA preservation, schedule content identity, fail-closed verification | `test_schedule_ir.py`, planned E2E-REAL-1 lit fixtures |
| Schedule→Tile boundary | Graph payload consumption, launch ABI materialization, numeric/layout/schedule preservation | `test_tile_ir.py`, planned `e2e_matmul_graph_schedule_tile.mlir` |
| CPU compiler path | supported op graph execution, artifacts for all compiler layers, eager fallback diagnostics | `test_end_to_end_matmul_cpu_path.py`, `test_transformer_compiler_example.py` |
| Runtime artifact launch | `JitFn.runtime_artifact()`, stable lineage, native image/descriptor joins, fallback-disabled launch, binding rejection, cleanup | `test_runtime_api_foundation.py`, `test_e2e_spine_orchestration.py` |
| Target profiles | GPU capability gates and lowering config attrs | `test_gpu_target.py`, `test_flash_attn_lowering.py` |
| Distributed planning | DP/TP/PP plans, pipeline stages, collective insertion preconditions | `test_distributed_plan.py`, `test_pipeline_stage_insertion.py`, `test_gpu_collective_insertion.py` |
| Reliability/runtime contracts | diagnostics, shape inference, runtime ABI, replay/QA helpers | `test_error_reporter.py`, `test_shape_inference.py`, `test_runtime_abi.py`, `test_qa_reliability_foundation.py` |

Unit tests should stay deterministic, CPU-only, and cheap enough for local edit-test loops.

## Performance Test Matrix

| Compiler/Runtime Concern | Required Performance Coverage | Gate |
| --- | --- | --- |
| JIT compile latency | Decoration plus Graph/Schedule/Tile/Target artifact construction for a transformer-shaped graph | Median wall time under `0.25s` on local CI-class CPU |
| Generated artifact size | Graph/Schedule/Tile/Target text remains compact for representative op graphs | Total text under `20 KiB` for the transformer proxy |
| GEMM roofline model | Model-shape and square GEMMs produce positive TFLOPs and expected compute/memory transitions | Large square GEMM compute-bound; tiny GEMM memory-bound |
| Attention roofline model | Flash-attention proxy reports positive tokens/sec, TFLOPs, and bounded MFU | `0 < mfu <= 1.0` |
| Collective model | All-reduce/reduce-scatter/all-gather model reports monotonic bus bytes and bounded utilization | `0 < utilization <= 1.0` |
| Benchmark suite schema | Combined benchmark JSON contains summary and all result families | Required keys present with non-empty result lists |

Performance tests should avoid requiring accelerators by default. Hardware-backed benchmarks belong in specialized GPU jobs and should be marked separately from deterministic proxy tests.

## Ordered Test-Implementation Backlog

1. Land E2E-REAL-0's lineage schema tests without changing backend selection.
2. Build the Schedule dialect's parse/print/verify and production-registration
   fixtures before changing either boundary pass.
3. Add the static matmul Graph/Schedule→Tile positive and negative lit pair.
4. Add one shared fixture/manifest consumed by the x86 and ROCm package tests;
   reject backend-authored Graph resynthesis on the canonical compiler route.
5. Add the Zen 5 AVX-512 and gfx1151 exact-device correctness commands and
   packet schema for the vertical slice. Keep AMX on its separate release gate.
6. Extend the existing gfx1151 and x86 performance ratchets with lineage
   identities and explicit promotion verdicts.
7. Parameterize the proven harness over softmax/reduction, attention, and then
   stateful/training families; route NVIDIA and Apple device proof to their
   owning hosts.
8. Add compile-time regression thresholds for larger transformer blocks only
   after multi-op lineage is real.
9. Promote context graph, spec, docs, and sample checks from unit guard tests
   into `tests/evals/` manifests once the generated-output workflow stabilizes.
