---
last_updated: 2026-07-18
audit_role: plan
plan_state: open
owner: shared compiler and backend owners
sync_key: E2E-SPINE-2026-07-18
---

# Backend end-to-end MLIR compilation plan

This is the authoritative execution plan for joining Tessera's frontend,
four-layer IR stack, architecture-owned code generators, native-image
packaging, and runtime launch paths. Generated dashboards remain status truth
for counts and per-op execution state; this plan owns the architectural seam,
work-item order, dependencies, and closure gates.

The plan covers Apple GPU, Apple CPU, x86, ROCm, and NVIDIA. It does not make
physical schedules portable. Logical operations, typed ABI contracts, artifact
schemas, diagnostics, and proof structure are shared; fragments, instructions,
memory movement, launch geometry, and performance selection remain target-owned.

## 1. Success definitions

The previous version used one overloaded meaning of E2E. This rebuild separates
three levels:

| Level | Definition | What it proves |
|---|---|---|
| **A — native execution** | A runtime path launches native code and matches an oracle. | The operation runs on the target. |
| **B — compiler-native seam** | Typed frontend/Tile IR lowers through typed Target IR without Python constructing Target-IR text or choosing a backend ABI symbol. | Decision #19 covers the Tile→Target boundary. |
| **C — canonical compiler E2E** | `canonical_compile` selects one exact-target registered pipeline, produces a native image plus launch descriptor, and `runtime.launch` consumes that descriptor without rediscovering the route. | One Graph→Schedule→Tile→Target→image→launch spine owns the execution. |

This plan closes a lane only at **Level C**. Existing Level-A execution remains
valuable and must stay available as a comparison candidate while migration is
in progress. Level B is the mandatory intermediate acceptance gate.

## 2. Verified baseline and structural gaps

The central gap is not a lack of kernels. Most native breadth already exists in
runtime libraries and per-backend emitters. The gap is that the canonical
compiler artifact stops at text and metadata, while Python reconstructs the
remaining target, packaging, or launch decision.

### 2.1 Shared compiler and runtime

- `python/tessera/compiler/driver.py::PIPELINE_BY_TARGET` maps every NVIDIA
  target to `tessera-lower-to-gpu`, maps Apple targets to the artifact
  pipelines rather than their executable `-runtime` pipelines, and represents
  ROCm as one family pipeline rather than an exact-architecture selection.
- `CompileResult` carries Graph, Schedule, Tile, and Target IR text plus gate
  metadata. `RuntimeArtifact` carries the same text, arbitrary metadata, and an
  ABI signature. Neither has a typed native-image or launch-descriptor field.
- `runtime.launch` resolves `(target, compiler_path)` through the execution
  matrix and `_executor_table()`. This is honest Level-A provenance, but for
  many lanes it lets runtime routing substitute for compiler-owned packaging.
- Pipeline registry descriptions and C++ builders are not fully isomorphic.
  A registry entry can therefore describe a target-specific pass that the
  registered C++ alias does not actually run.

### 2.2 NVIDIA

NVIDIA currently has two registered but disconnected families:

1. `src/transforms/lib/Passes.cpp::buildCUDA13Pipeline` owns the shared
   Graph→Tile path. Its sm90/sm100/sm120 aliases use the same builder, hardcode
   the `nvidia_sm90` control-flow guard, run WGMMA/TMA marker-oriented passes,
   and never call `LowerTileToNVIDIA(sm)`.
2. `NVIDIALowering.cpp` registers Tile→`tessera_nvidia`→NVVM pipelines. The
   Blackwell alias passes `sm=100`; no registered full pipeline passes
   consumer Blackwell `sm=120` even though the raw lowering supports it.

Actual sm120 execution is broad through NVRTC/PTX and shipped CUDA ABIs, but
those routes are selected from Python rather than consuming compiler-produced
NVVM/PTX packages. Marker-only WGMMA, TCGEN05, TMA, mbarrier, and control forms
remain distinct from the proven sm120 `mma.sync`/NVFP4 lowering.

### 2.3 ROCm

`buildTesseraROCMBackendPipeline` wires the WMMA GEMM generator, Wave/LDS
planning and legality, Tile→ROCm, kernel ABI lowering, and ROCDL lowering. About
70 additional `GenerateROCM*KernelPass` implementations are registered only as
standalone passes. They consume typed `tessera_rocm.*` directives, but those
directives are usually authored as MLIR text by `runtime.py` before invoking a
single-generator `tessera-opt` pipeline. The gfx1151 native breadth is Level A;
GEMM is the main Level-B lane.

### 2.4 x86

`tessera-lower-to-x86` is a real registered Graph→native-call pipeline, but
`TileToX86Pass` covers only matmul, fused epilogue, and an artifact-only KV
shape. The broad AVX-512 runtime library is selected by executor IDs. There is
no typed x86 Target dialect, so the hardware-free Target-IR contract is missing
for most native x86 families.

### 2.5 Apple

Apple GPU is closest to the target model. The registered
`tessera-lower-to-apple_gpu-runtime` pipeline runs longest-fusion-first lowering
and emits typed `tessera_apple.gpu.kernel_call` values with concrete symbols.
However, canonical pipeline selection still chooses
`tessera-lower-to-apple_gpu`, the artifact pipeline. GA, EBM, linalg, PPO, and
other value-mode routes also remain outside the registered runtime pipeline.

Apple CPU's executable registered pipeline lowers only rank-2 f32 matmul to
`cblas_sgemm`; canonical selection chooses its artifact pipeline.

## 3. Target architecture

```text
Python API / @jit
        │
        ▼
Graph IR
        │  shared canonicalization, fusion, effects, shape and dtype legality
        ▼
Schedule IR
        │  distribution, placement, pipeline and residency intent
        ▼
Tile IR
        │  logical fragments, memory movement, async/control contracts
        ▼
typed backend Target IR
        │  architecture-owned instructions, ABI, launch and workspace contract
        ▼
NativeImageArtifact
        │  PTX/cubin, HSACO, ELF/object/shared image, metallib/MSL package
        ▼
LaunchDescriptor
        │  image digest, entry point, ordered bindings, scalars, grid/workspace
        ▼
runtime launcher registration
        │
        ▼
exact-device execution and oracle
```

### 3.1 Ownership boundary

Python may orchestrate compilation processes, content-addressed caching,
candidate measurement, and launch submission. It may not:

- construct backend Target-IR directive text from operation metadata;
- select an ABI symbol that was not emitted in the compiler launch descriptor;
- infer an exact architecture after Target IR was generated;
- label a reference or side-channel result compiler-native;
- turn a missing compiler package into a silent Level-C fallback.

The runtime may retain the execution matrix and executor table for legacy and
Tier-3 candidates. A Level-C compiler candidate enters that table through one
generic descriptor-consuming executor, not one new hand-authored executor ID
per operation.

### 3.2 Native image contract

`NativeImageArtifact` is the content-addressed output of the compiler/plugin
boundary. Its schema must include:

- schema version, exact target and architecture;
- registered pipeline name and compiler/toolchain fingerprint;
- Target-IR digest and native payload digest;
- binary format (`ptx`, `cubin`, `hsaco`, `elf`, `shared_object`, `metallib`,
  or an explicitly registered equivalent);
- one or more entry points with stable ABI identifiers;
- compile state (`cold`, `warm_cache`, `prepackaged`) and cache key;
- optional measured/inspected resource record with provenance.

The portable schema stores no CUDA warp map, AMD wave layout, Metal
threadgroup schedule, or x86 vector width. Those remain payload/Target-IR data.

### 3.3 Launch descriptor contract

Each entry point carries a typed `LaunchDescriptor`:

- image digest and entry symbol;
- ordered buffer bindings with direction, dtype, rank, and layout requirements;
- scalar arguments and dynamic-shape guards;
- grid/block/threadgroup geometry or a registered runtime-computed policy;
- dynamic shared/LDS/threadgroup-memory and workspace requirements;
- stream/queue ordering, residency, and synchronization semantics;
- provenance fields used by `runtime.launch` and the execution matrix.

Malformed bindings, unsupported shapes/dtypes, stale images, and missing
launchers must fail with registered diagnostics before device submission.

## 4. Work register

Statuses in this table describe this plan, not generated execution counts.

| Order | ID | Priority | State | Scope | Completion gate |
|---:|---|---|---|---|---|
| 1 | **E2E-SPINE-0** | P0 | closed | Truth and ownership foundation | `tessera.target_pipeline.v1` makes every capability target total, derives the behavior-preserving driver map, records declared/shared/unsupported ownership, reconciles shared NVIDIA builder metadata, and generates Level-A/B/C inventory. |
| 2 | **E2E-SPINE-1** | P0 | closed | Native-image and launch-descriptor schema | `tessera.native_image.v1` and `tessera.launch_descriptor.v1` provide deterministic serialization, payload/image/descriptor hashes, cache fingerprints, ordered ABI validation, shape/workspace/ordering contracts, and registered stale/malformed diagnostics without backend schedules. |
| 3 | **E2E-SPINE-2** | P0 | closed | Canonical orchestration | `CompileResult`/`RuntimeArtifact` carry compiler-produced image and launch data; the driver records each stage and a generic runtime executor validates and consumes the descriptor through exact-target hooks. Artifact-only and legacy candidates remain explicit and descriptor routes never fall back. |
| 4 | **NVIDIA-E2E-1** | P0 | queued | sm120 vertical slice | A registered exact-sm120 Graph→Tile→`tessera_nvidia`→NVVM→PTX pipeline packages and launches f16 plus NVFP4 matmul through `tessera_nvidia_ptx_register/invoke`, with no Python kernel synthesis or ABI rediscovery. |
| 5 | **ROCM-E2E-1** | P0 | queued | Typed directive pilot | One non-GEMM lane (softmax first) lowers from typed frontend/Tile IR to a typed `tessera_rocm.*` directive, existing generator, ROCDL, HSACO, and launch descriptor. Runtime text synthesis is removed for that lane only after parity. |
| 6 | **APPLE-E2E-1** | P1 | queued | Canonical Apple GPU spine | Canonical compilation selects the executable Apple GPU pipeline; GA/EBM/linalg/PPO value-mode families receive registered typed lowering or explicit unsupported states. Existing Metal selectors and schedules do not change without Apple evidence. |
| 7 | **X86-E2E-1** | P1 | queued | Typed x86 breadth | Introduce the hardware-free x86 Target contract and migrate softmax/reduction first, followed by existing stable-ABI families. Direct executor-only routing retires per lane after equivalence. |
| 8 | **ROCM-E2E-2** | P1 | queued | Directive/generator breadth | Expand the proven emitter pattern by semantic families; wire only generators whose typed producers, legality, ABI, and exact-device proof exist. Do not append all generators blindly to one pass list. |
| 9 | **NVIDIA-E2E-2** | P1 | queued | Native lowering breadth and per-SM split | Promote real NVVM/PTX lowering by family, separate sm90/sm100/sm120 builders, and replace marker operations only with matching ISA/toolchain proof. sm90/sm100 execution remains exact-device gated. |
| 10 | **APPLE-CPU-E2E-1** | P2 | queued | Apple CPU breadth | Extend the executable Accelerate/LAPACK pipeline beyond rank-2 f32 matmul and make canonical selection honest per supported family. |
| 11 | **E2E-SPINE-3** | P2 | queued | Fleet proof and closeout | Cross-backend differential fixtures, generated Level-A/B/C dashboard truth, cache reproducibility, and per-backend release packets demonstrate the completed migrated scope. |

## 5. Dependency and landing strategy

```text
E2E-SPINE-0
      │
      ▼
E2E-SPINE-1 ──► E2E-SPINE-2
      │                │
      ├────► NVIDIA-E2E-1 ──► NVIDIA-E2E-2
      ├────► ROCM-E2E-1   ──► ROCM-E2E-2
      ├────► APPLE-E2E-1  ──► APPLE-CPU-E2E-1
      └────► X86-E2E-1
                       │
                       ▼
                 E2E-SPINE-3
```

The foundation PRs are host-free and may land before exact-device follow-ups.
Each backend vertical slice is independently reviewable and retains the current
Level-A route until Level-C correctness, provenance, and performance
non-regression are proven. A backend cannot block an unrelated backend's
vertical slice once the shared schema is stable.

## 6. Backend engineering notes

### 6.1 NVIDIA first slice

Build one new registered `sm=120` composition rather than extending the shared
alias with conditionals. It must run the shared Graph→Tile passes, then
`LowerTileToNVIDIA(120)`, `LowerNVIDIAToNVVM`, NVVM translation, PTX packaging,
and the existing PTX launch bridge. Start with the already-proven f16 and NVFP4
matmul contracts so failures isolate the spine rather than kernel semantics.

The existing shipped CUDA ABI and NVRTC candidates remain Tier-3/Tier-2 peers.
No selector changes during spine closure. The first slice succeeds when the
compiler-produced candidate is independently correct and measurable.

### 6.2 ROCm first slice

Do not wire all registered generators into the default pipeline. First add one
typed directive-emitting pass for softmax because it exercises a non-GEMM
operation, reduction semantics, dynamic width guards, kernel ABI, ROCDL, and
the existing exact-gfx1151 launcher without importing attention complexity.

After the pilot, group generators by shared typed producer and ABI shape:

1. elementwise/reduction/norm;
2. attention and reasoning attention;
3. structured, sparse, and MoE;
4. recurrent/SSM and control;
5. linalg, optimizer, loss, GA, and EBM.

Each group gets a totality table. Missing producers or unsupported target forms
remain named diagnostics, not no-op generator passes.

### 6.3 Apple

Apple GPU already demonstrates typed kernel-call Target IR. Its first task is
canonical-driver and pipeline-totality repair, followed by registering the
currently unselected value-mode families. This work must not collapse MPS,
MPSGraph, synthesized MSL, simdgroup, Metal 4, or packaged-subgraph candidates
into one physical route.

Apple CPU remains a separate target. GPU progress cannot promote CPU rows.

### 6.4 x86

Use the existing stable AVX-512 C ABIs as implementation vehicles, but create a
typed Target contract before adding broad `TileToX86` calls. The pilot should
prove that Target IR carries the selected ABI and bindings and that the generic
launch descriptor reaches the same native function as the legacy executor.

## 7. Proof contract for every migrated lane

Every lane moving from Level A to Level C requires four layers:

1. **Host-free typed proof**
   - Graph→Schedule→Tile→Target pipeline runs from a registered exact-target
     name;
   - verifier negatives cover dtype, layout, shape, binding, workspace, and
     unsupported target forms;
   - golden Target IR and launch-descriptor serialization are deterministic.
2. **Native-image proof**
   - the exact toolchain accepts the compiler output;
   - PTX/SASS, AMDGCN/HSACO, Metal package, or x86 object inspection confirms
     the intended entry and architecture;
   - compiler and payload fingerprints are retained.
3. **Exact-device proof**
   - the descriptor-consuming launcher runs the image;
   - output matches the shared oracle on aligned, ragged, boundary, and invalid
     contracts applicable to the family;
   - forced descriptor/image/launcher failure cannot earn native provenance;
   - device allocations, streams/queues, workspaces, and caches clean up.
4. **Measured non-regression**
   - compiler-produced and retained candidates run in isolated serial trials;
   - device and end-to-end timing remain separate;
   - cold/warm compile and cache state plus resources are retained;
   - a production selector changes only through the backend's existing stable
     promotion policy.

Exact-device proof never transfers between architectures. Host-free Target IR
may be authored anywhere, but NVIDIA, ROCm, Apple, and x86 execution claims are
owned by their corresponding hosts.

## 8. Registry and lifecycle gates

Every E2E PR must assess and update all applicable surfaces:

- exact target capability and pipeline registry;
- pass registration and metadata;
- backend manifest and execution matrix;
- runtime ABI/launcher registry;
- diagnostic-code registry;
- test-coverage and conformance registries;
- generated dashboards;
- Apple, NVIDIA, and ROCm plans using sync key `E2E-SPINE-2026-07-18`.

`artifact_only`, `compileable`, `packaged`, `launchable`, and exact-device
verified states must remain distinct. A native image without a launcher is not
executable; a launcher without an execute/compare fixture is not device proof;
Level-A execution outside the compiler spine is not Level C.

## 9. Non-goals

- No common physical fragment ABI across CUDA, ROCm, Metal, and x86.
- No removal of hand-tuned or shipped kernels; they remain permanent arbiter
  candidates.
- No mass selector promotion as a side effect of compiler plumbing.
- No requirement that every primitive have a native kernel on every target.
- No giant multi-hardware PR. Shared contracts land first; exact-device
  follow-ups are linked by the synchronization key.
- No Python ban. Python remains a valid orchestration layer after typed Target
  IR and compiler-owned launch metadata exist.

## 10. Closure definition

The plan can move to `closed` and then archive only when:

1. every supported exact target resolves through one canonical registered
   pipeline or one explicit terminal state;
2. native image and launch descriptor schemas are versioned, serialized,
   validated, and consumed by a generic runtime path;
3. NVIDIA sm120 and ROCm gfx1151 each have at least one nontrivial Level-C
   exact-device lane, with their breadth queues explicitly classified;
4. Apple GPU canonical compilation uses its executable typed pipeline for its
   supported scope;
5. x86 has a hardware-free Target contract and at least one migrated non-GEMM
   Level-C lane;
6. Apple CPU breadth is either implemented for the declared scope or closed
   with explicit reference/unsupported classifications;
7. generated dashboards distinguish Levels A, B, and C without prose-inferred
   counts;
8. all backend plans and release evidence agree, and no live follow-on points
   to this plan as unfinished.

## 11. Foundation landing record

**E2E-SPINE-0 is complete (2026-07-18).** It is deliberately
behavior-neutral:

1. `TARGET_PIPELINE_RESOLUTIONS` covers every canonical capability target and
   names the current driver pipeline separately from the declared C++ pipeline;
2. `PIPELINE_BY_TARGET` is derived from that registry, preserving every prior
   production route including exact-ROCm artifact fallback;
3. the NVIDIA SM90/SM100/SM120 alias metadata now matches the actual shared
   `buildCUDA13Pipeline` pass list instead of claiming uncomposed tcgen05 work;
4. totality tests join capabilities, driver selection, registered pipelines,
   registration sources, and pass ownership;
5. `compilation_spine_inventory.{csv,md}` reports Level A/B/C directly from the
   target, pipeline, and execution registries;
6. no runtime executor, selector, code generator, or launch behavior changes.

### E2E-SPINE-1 landing record

**E2E-SPINE-1 is complete (2026-07-18).** The normative contract is
[`../../spec/NATIVE_ARTIFACT_SPEC.md`](../../spec/NATIVE_ARTIFACT_SPEC.md), and
the implementation is `python/tessera/compiler/native_artifact.py`.

- native images carry exact target/architecture, registered pipeline,
  compiler/toolchain identity, Target-IR and payload digests, binary format,
  entry ABI, compile/cache state, and optional provenance-bearing resources;
- launch descriptors carry an image/symbol/ABI join, one ordered argument
  space, dtype/rank/layout/alignment requirements, dynamic-shape guards,
  generic geometry, local memory, workspace, ordering, and residency;
- deterministic JSON round-trips recompute payload, image, descriptor, and
  cache fingerprints and reject corrupt or stale data;
- invocation validation rejects wrong buffers, scalars, shapes, layouts,
  alignments, symbols, and ABI identifiers before submission;
- five `E_NATIVE_IMAGE_*` / `E_LAUNCH_*` diagnostics are registered;
- no `CompileResult`, `RuntimeArtifact`, executor, selector, backend schedule,
  or device claim changes in this item.

### E2E-SPINE-2 landing record

**E2E-SPINE-2 is complete (2026-07-18).** It closes the portable join without
claiming a backend vertical slice:

- compiler bundles record Graph, Schedule, Tile, Target, backend, native-image,
  and launch-descriptor stages plus the highest honest orchestration state;
- bundle construction rejects image target, pipeline, or Target-IR drift and
  descriptor/image/symbol/ABI drift;
- `CompileResult` and `RuntimeArtifact` carry the typed objects directly, and
  deterministic runtime/AOT serialization validates nested and outer hashes;
- AOT cache identity and the versioned persistent-entry manifest join the
  lookup key, outer artifact hash, native-image key, and descriptor fingerprint;
- `runtime.launch` validates named bindings and shape contracts before an
  exact-target, binary-format-constrained submission hook;
- a missing hook is explicit `unimplemented`, and a descriptor route never
  falls through to a legacy executor or selector;
- no CUDA, HIP, Metal, or x86 launcher is registered by this item, so Level-C
  inventory remains absent until the architecture-owned vertical slices land.

The next software-actionable items are **NVIDIA-E2E-1**, **ROCM-E2E-1**,
**APPLE-E2E-1**, and **X86-E2E-1**, each with its own proof requirements.

## 12. Evidence routing

- Runtime/native Level-A truth:
  [`../generated/runtime_execution_matrix.md`](../generated/runtime_execution_matrix.md)
- Cross-registry Level-A/B/C truth:
  [`../generated/compilation_spine_inventory.md`](../generated/compilation_spine_inventory.md)
- Per-op E2E truth:
  [`../generated/e2e_op_coverage.md`](../generated/e2e_op_coverage.md)
- Target support:
  [`../generated/apple_target_map.md`](../generated/apple_target_map.md),
  [`../generated/rocm_target_map.md`](../generated/rocm_target_map.md), and
  [`../generated/nvidia_sm90_target_map.md`](../generated/nvidia_sm90_target_map.md)
- Compiler architecture:
  [`../compiler/COMPILER_THEORY_OF_OPERATION.md`](../compiler/COMPILER_THEORY_OF_OPERATION.md)
- Backend coordination:
  [`BACKEND_AUDIT.md`](BACKEND_AUDIT.md),
  [`apple/todo.md`](apple/todo.md),
  [`nvidia/todo.md`](nvidia/todo.md), and
  [`rocm/todo.md`](rocm/todo.md)
