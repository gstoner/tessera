---
last_updated: 2026-07-20
audit_role: plan
plan_state: landing
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

NVIDIA started this plan with two registered but disconnected families. The
NVIDIA-E2E-1 vertical slice and the first NVIDIA-E2E-2 landing have now joined
the SM120 lane and split target ownership:

1. `src/transforms/lib/Passes.cpp::addCUDA13PipelineForSM` owns the exact-SM
   Graph→Tile front pipelines. SM90 alone consumes the proven Hopper
   WGMMA/FlashAttention marker passes; SM100 and SM120 retain typed carriers for
   their target-owned backends. Async copies carry explicit completion tokens
   through waits and matrix consumers.
2. `NVIDIALowering.cpp` registers separate
   `tessera-lower-to-nvidia-sm90`, `-sm100`, and `-sm120` Tile→
   `tessera_nvidia`→NVVM pipelines. Exact execution remains gated to the named
   device; the split does not infer SM90/SM100 correctness from SM120.

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
`TileToX86Pass` still covers only a subset. X86-E2E-1 now lowers the shared
softmax and reduction Tile envelopes to typed stable-C-ABI calls, packages the
AVX-512 shared object as the native image, and supplies a canonical launch
descriptor. Other families remain selected by executor IDs and therefore do
not yet have the same hardware-free Target-IR contract.

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
| 4 | **NVIDIA-E2E-1** | P0 | closed | sm120 vertical slice | A registered exact-sm120 Graph→Tile→`tessera_nvidia`→NVVM→PTX pipeline packages and launches f16 plus NVFP4 matmul through `tessera_nvidia_ptx_register/invoke`, with no Python kernel synthesis or ABI rediscovery. |
| 5 | **ROCM-E2E-1** | P0 | closed | Typed directive pilot | One non-GEMM lane (softmax first) lowers from typed frontend/Tile IR to a typed `tessera_rocm.*` directive, existing generator, ROCDL, HSACO, and launch descriptor, with exact-device correctness and measured parity against the retained route. Runtime text synthesis remains a separate retirement decision. |
| 6 | **APPLE-E2E-1** | P1 | closed / bounded Level C | Canonical compilation selects the executable Apple GPU pipeline and defaults its bounded static descriptor scope to hash-bound native images plus descriptors: rank-2 f32 softmax/transpose; f32/f16/bf16 rank-3 batched GEMM; strict and mask/reference-KL/entropy PPO; EBM energy/Langevin/refinement/partition; cl30 Clifford geometric product; and static/batched Cholesky, Cholesky-solve, and triangular-solve. Per-family package/oracle/replay proof passes on the exact Apple device from the owned rebuilt dylib. Composite, dynamic, stateful, unsupported, and multi-result GPU routes, plus fleet/second-device proof, are explicitly APPLE-NATIVE-E2E-2 work. Existing Metal selectors and schedules do not change without Apple evidence. |
| 7 | **X86-E2E-1** | P1 | closed | Typed x86 pilot and selector closure | Typed C-ABI Target IR and canonical descriptors cover f32 softmax, last-axis sum/mean/max, rank-2 matmul, plain MHA, and bias/window/softcap MHA. Exact-host correctness and retained-route performance gates pass. Canonical compilation defaults eligible static modules to the descriptor route; explicit opt-out and unsupported contracts remain on retained routes. |
| 8 | **X86-E2E-2** | P1 | closed | AVX-512 stable-ABI breadth | All 76 exports are inventoried. The 33 cohort-3/4 ABIs have total typed/effect/status contracts and selector dispositions. Exact-host Graph descriptors promote measured flat gather, unreduced pointwise loss, Cholesky, and triangular solve domains; 29 non-isomorphic, composite, stateful, multi-output, or specialized entries retain their explicit routes. |
| 9 | **ROCM-E2E-2** | P1 | closed | Directive/generator breadth | Reduction f16/bf16/f32 input to f32 output passes all nine comparable-device/E2E gates. Direct paged-KV and MoE dispatch have typed f32/i32 descriptors, negative/exact-gfx1151 evidence, and measured non-winning dispositions that retain their production routes. |
| 10 | **NVIDIA-E2E-2** | P1 | closed | Native lowering breadth and per-SM split | SM120 semantic breadth and measured route dispositions are complete. SM90/SM100 and exact multi-GPU proof have formal hardware-deferred terminals and must reopen as separate exact-device follow-ups. |
| 11 | **APPLE-CPU-E2E-1** | P2 | closed / bounded Level C | Apple CPU static linalg breadth | Static f32 rank-2 matmul/gemm, rank-3 BMM, single-result Cholesky/triangular-solve/Cholesky-solve, and tuple-output LU/QR/SVD descriptors pass exact-host execute/compare and replay through the owned rebuilt dylib. Dynamic shapes, other dtypes, and non-linalg contracts are explicitly APPLE-NATIVE-E2E-2 work. |
| 12 | **E2E-SPINE-3** | P2 | landing | Fleet proof and closeout | Family-granular cross-backend differential fixtures, generated Level-A/B/C dashboard truth, cache reproducibility, and hash-sealed per-backend release packets demonstrate only the completed migrated scope. |
| 13 | **APPLE-NATIVE-E2E-2** | P3 | landing / local scope complete | Apple native breadth and fleet proof beyond the bounded E2E-1 contracts | The bounded local descriptor program is complete on the exact Apple7 Metal-4 device. In addition to static rank-2 f16/bf16 CPU matmul/gemm, the descriptor-state registry and exact-host replay proof admit static rank-2 f32 CPU row-softmax. GPU GELU packages static and dynamic f32/f16/bf16 storage with explicit fp32-accumulation provenance, two-byte low-precision bindings, independent storage-rounding oracles, and rank/dtype/result-shape/scalar rejection ratchets. Ordered reduced GPU SVD and ReplaySSM lifecycle descriptors are also complete. Three further explicit dynamic GPU contracts are packaged: rank-1 i32 popcount with `Elements`, rank-2 f32 last-axis count-nonzero with `Outer/AxisExtent`, and rank-2 f32 row-softmax with `Rows/Columns`. Ordered top-k uses a dedicated status-returning Metal ABI rather than MPSGraph: numeric values descend, NaNs sort last, and lower indices win ties; `(values,indices)` bindings, `Rows/Columns/K`, exact-device oracle, replay, and rejection are proven. Metal-4 composite execution is sealed by tree digest, reflected positional externals, a private intermediates heap, and replay-safe cache identity. Two independent 50-repetition by 5-trial reports are sealed in the isolated strict-v2 `package_subgraph` ledger: the package is admitted at `64x64x64`, the live route is retained at `256x256x256`, and device-domain rows remain explicitly ineligible because only complete-call timing is comparable. The local CPU audit finds no further owned static non-linalg ABI, and no speculative wrapper is admitted. Independently sealed second-device/fleet evidence is the only remaining hardware terminal; retained/reference routes require a separately owned future ABI item. |

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
      └────► X86-E2E-1 ──► X86-E2E-2
                       │
                       ▼
                 E2E-SPINE-3
```

The foundation PRs are host-free and may land before exact-device follow-ups.
Each backend vertical slice is independently reviewable and retains the current
Level-A route until Level-C correctness, provenance, and performance
non-regression are proven. A backend cannot block an unrelated backend's
vertical slice once the shared schema is stable.

### 5.1 E2E-SPINE-3 proof architecture

Fleet closeout is a join over immutable evidence, not a new portability layer.
The join key is `(fixture_id, operation_family, dtype, shape, semantic_contract)`;
the physical schedule and native payload remain backend-owned. Each backend run
produces a machine-readable report with:

- exact target, device identity, source commit, compiler/toolchain fingerprint,
  and the bounded operation-family scope being claimed;
- explicit Level-A/B/C provenance for each fixture, including native-image and
  launch-descriptor digests for Level C;
- values or comparison metrics against the shared oracle, using the fixture's
  registered numerical policy rather than a fleet-wide tolerance;
- cold and warm cache keys plus image/descriptor digests, so reproducibility is
  proven without requiring cold and warm compile latency to be equal;
- benchmark rows with a declared target-appropriate kernel domain
  (`device_event` or `kernel_wall`) plus end-to-end timing, repetitions,
  warmup/discard policy, selected route, resource fingerprint, and stability
disposition.

Packet identity is separately keyed by `(target, architecture)`. This is
required even when two architecture envelopes share the same compiler target:
`x86_64_base` and `x86_64_avx512` are independent packets, just as `sm_120a`
cannot satisfy `sm_90a` or `sm_100a`. The NR2 WSL host owns non-AVX512
`x86_64_base` plus `sm_120a`; Strix Halo owns `x86_64_avx512` plus `gfx1151`;
the M1 Max owns Apple CPU plus Apple7 GPU proof. Evidence never transfers
between those hosts or architectures.

A release packet hash-seals that report and its referenced evidence files.
Portable CI validates schemas, hashes, registry joins, and generated dashboard
drift. Exact-device hosts produce numerical and timing evidence locally; CI must
not relabel an unavailable device as passed or inherit evidence between exact
architectures. Hardware-deferred terminals remain first-class dashboard rows.

The generated fleet dashboard is family-granular. Target-wide `partial` or
`absent` inventory cells remain useful ownership truth but cannot erase a
bounded Level-C family packet; conversely, one family packet cannot promote the
whole target. A packet is release-ready only when all fixtures in its declared
scope pass, cold/warm identities reproduce, required benchmark domains are
present, and every referenced file matches the sealed manifest.

The first exact-host recorder slice implements portable x86 f32 softmax and
last-axis sum through compiler-selected `x86_64_base` C-ABI symbols in a shared
image compiled with `-march=x86-64 -mno-avx -mno-avx2 -mno-avx512f`. Unsupported
base envelopes reject in the Tile-to-x86 pass instead of falling through to an
AVX512 symbol. The hash-sealed WSL packet now binds its shared numerical
fixtures, prepackaged image/descriptor replay, resource fingerprints, and
repeated-median `kernel_wall` plus end-to-end rows to landed source commit
`9f3757ef2dda2dd61ff94f1aefe0244f1b80f064`; every row passes the 4% gate and
disassembly contains no YMM/ZMM use. The independent SM120 packet binds the
same shared softmax/reduction fixtures, cold/warm compiler-cache identity,
ptxas resource fingerprints, and repeated-median device-event plus end-to-end
rows to that commit on the RTX 5070 Ti. All four SM120 timing rows pass the
unchanged 4% gate. The generated dashboard marks only these four bounded
families release-ready; broader SM120, AVX512, gfx1151, and Apple packet scope
remains pending.

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

X86-E2E-1 closed that pilot for static f32 softmax and last-axis f32
sum/mean/max reduction. The registered pass emits typed `func.call` operations
for the selected ABI, while the native image owns the shared-object bytes and
the descriptor owns ordered buffers and scalar dimensions. The generic runtime
loads the image from an in-memory file and reports CPU wall time. Exact-host
correctness covers scalar, odd, and vector-width-crossing shapes; the six-row
alternating comparison is within the 10% non-regression bound against retained
`x86_softmax_compiled` and `x86_reduce_compiled` routes. Those retained routes
remain selected until the wider stable-ABI migration and canonical-selector
decision are complete.

The next three slices consume the existing `tile.matmul_kernel` and
`tile.attention_kernel` carriers. Static rank-2 f32 matmul maps to
`tessera_x86_avx512_gemm_f32`; static rank-4 f32 MHA maps to
`tessera_x86_flash_attn_f32`; and exact-shape bias, symmetric local window, or
logit softcap selects `tessera_x86_flash_attn_ext_f32`. The descriptor boundary
deliberately rejects GQA/MQA head expansion and dropout because the shipped C
ABI does not own those transformations. Three matmul shapes and basic/extended
attention execute against numerical or retained-route oracles. The paired
21-trial record passes the unchanged 10% end-to-end bound: matmul spans
0.930--0.946x, plain attention is 0.996x, and extended attention is 0.985x.
For these five static contracts, `canonical_compile` now selects native
packaging automatically when the matching compiler and shared image are
available. The typed descriptor is therefore the canonical route and the
runtime does not rediscover `x86_softmax_compiled`,
`x86_matmul_family_compiled`, or `x86_flash_attn_compiled`. Explicit
`package_native=false`, dynamic or unsupported contracts, and `@jit` calls
whose output/scalar bindings have not yet been specialized retain their prior
route. This is scoped selector retirement, not deletion of the comparison
executors.

X86-E2E-2 owns the remaining AVX-512 breadth. Its first deliverable is a total
symbol-to-operation/ABI inventory, because the shared library mixes direct
one-operation kernels with stateful or host-orchestrated compositions. Direct
families are migrated in these cohorts:

1. unary, binary, compare, predicate, logical, bitwise, and where;
2. scan, arg-reduction, normalization, and positional encoding;
3. gather/scatter/sort, FFT/spectral, sparse, and linalg;
4. loss, quantization, optimizer, MoE, SSM/DeltaNet, EBM, and RNG/stateful
   families.

Each cohort needs an operation-total typed producer, stable ABI and descriptor,
negative contracts, exact-host numerical proof, and two-domain comparison
before any selector change. Composite host programs remain explicit
compositions unless one stable native entry point actually owns their complete
semantics; the existence of an AVX-512 object file alone is not Level C.

The first X86-E2E-2 cohort is landing under sync key
`X86-E2E2-ELEMENTWISE-2026-07-20`. The total export inventory is
[`X86_AVX512_ABI_INVENTORY.md`](X86_AVX512_ABI_INVENTORY.md): 76 symbols split
into 31 AVX-512 direct entries, 19 reference entries, and 26 other direct or
specialized entries. `tile.elementwise_kernel` now carries the unary, binary,
and predicate semantics without embedding an x86 symbol; `TileToX86Pass`
lowers it to three stable C ABIs, native packaging emits typed descriptors, and
the descriptor launcher executes caller-owned f32/bool outputs. Exact-host
tests cover all 9 unary, 8 binary, and 3 predicate kinds.

The 41-trial retained comparison records unary speedups of 0.970--3.306x,
binary 0.836--2.803x, and predicate 0.985--5.837x. The two small binary rows
miss the 10% bound by 4--6 microseconds of fixed descriptor cost, while the
focused sweep first passes at 16,384 elements. Canonical compilation therefore
promotes every valid static unary/predicate request and binary requests at or
above 16,384 elements. Smaller binary requests retain
`x86_binary_compiled`; explicit descriptor packaging remains supported. This
is a measured partial selector promotion, not deletion of comparison routes.

The next three X86-E2E-2 slices land under sync key
`X86-E2E2-TYPED-LOGIC-2026-07-20`. The same typed carrier now represents six
f32-to-bool comparisons, four bool logical operations, and five i32 bitwise
operations including unary `popcount`. Verifier rules pin logical i8 physical
storage, bitwise i32 storage, result storage, and unary/binary arity. The x86
capability registry now reflects the already-shipped bool and int32 ABIs rather
than rejecting their canonical Graph IR requests. Typed lowering supplies a
null second pointer for the shipped unary logical/bitwise C ABI, and descriptors
retain the logical bool or int32 binding contract.

All 15 operation kinds pass exact-host numerical comparison, including ordered
NaN comparison semantics and signed bit-pattern/popcount cases. The committed
41-trial record uses binary representatives and four sizes per ABI. Compare
speedups span 0.766--2.229x, logical 0.966--8.221x, and bitwise
0.835--1.810x. Logical is promoted for every valid static shape. Compare and
bitwise retain their legacy routes below the conservative 32,768-element floor;
both descriptor routes pass the 10% bound at that threshold and above. Explicit
typed packaging remains available below the selector threshold.

The remaining flat cohort lands under sync key
`X86-E2E2-FLAT-FOLLOWON-2026-07-20`. Typed carriers and descriptors now cover
where, the 21-kind transcendental family, pow, and SiLU-multiply. The retained
21-trial comparison promotes transcendental at every valid static extent,
pow/SiLU-multiply from 8,224 elements, and where only at the directly measured
1,048,576-element winner. Smaller shapes retain their existing compiled routes.

The datatype slice lands under sync key `X86-E2E2-DTYPE-2026-07-20`.
`x86_dtype_contract.py` separates storage, CPUID requirements, compute and
accumulator types, and Tessera readiness. The Ryzen AI Max+ 395 profile no
longer advertises AMX. BF16-to-FP32 uses AVX512_BF16, U8/S8-to-S32 uses
AVX512_VNNI, and FP64 uses AVX512F/FMA. Exact-host aligned, ragged, and larger
rows match references; native kernel speedups span 1.283--12.212x. FP8 remains
software-emulated/planned, and the three new datatype descriptors remain
explicit rather than automatically selected until a production-route policy is
measured.

Cohorts 3 and 4 close under sync key
`X86-E2E2-BREADTH-2026-07-20`. A target-owned, deliberately side-effecting
`tile.x86_abi_kernel` carries the exact versioned C ABI after any required host
packing. Its verifier requires the `tessera_x86_` symbol namespace, a
`tessera.x86.*.v1` ABI ID, explicit family/effect ownership, explicit status
return policy, and an operand list restricted to the supported C scalar and
pointer types. `TileToX86Pass` derives the external function declaration from
those typed operands; Python no longer reconstructs a signature at launch.

The cohort registry is total for 12 cohort-3 and 21 cohort-4 exported entries.
It records in-place FFT/sort/scatter, multi-output factorization/optimizer/SSM,
status-returning KV mutation, FP16/BF16 SSM storage, and Philox/EBM state
inputs without marking them pure. ABI-shaped restrictions stay explicit:
SDDMM consumes transposed RHS storage, Clifford consumes blade-major storage,
bitonic sort is host padded, and the direct FFT entry is radix-2 C2C. Exact
Ryzen-host descriptor tests prove the complete export inventory and exercise
direct, Graph-level, factorization/solve, and stateful representatives,
including KV append status and mutation.

`benchmarks/baselines/x86_avx512_e2e_cohort34_comparison.json` records 21
serial-alternating retained/descriptor trials over three domains for each
isomorphic Graph family. The first rows meeting the unchanged 10% bound are
1,048,576 gather outputs, 16,384 unreduced loss elements, 2,048 Cholesky output
elements, and 512 triangular-solve output elements. Canonical compilation
promotes exactly those domains. The same record contains one disposition for
every cohort ABI: the other 29 remain `retain_composite` or
`retain_specialized` because their public contract needs packing, composition,
state, multiple results, or semantics narrower than the exported entry. A
performance comparison is not applicable where there is no semantically
equivalent descriptor candidate. This operation-total decision closes
X86-E2E-2 without relabeling composite work as direct Level C.

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

### NVIDIA-E2E-1 and NVIDIA-E2E-2 landing record

**NVIDIA-E2E-1 is complete (2026-07-19).** Canonical SM120 compilation owns
typed f16 and general-shape NVFP4 matmul images and launch descriptors, lowers
through the registered NVIDIA/NVVM/PTX path, validates with `ptxas`, and
registers/submits through the shipped PTX bridge. Exact RTX 5070 Ti aligned,
ragged, boundary, and non-origin scale rows compare to their numerical oracles;
image identity, cold/warm state, and resource evidence are retained without a
selector change.

**ROCM-E2E-1 is complete (2026-07-19).** The first slice adapts the shared
`tile.softmax_kernel(X, O, Rows, K)` envelope to a typed
`tessera_rocm.softmax` directive in `lower-tile-to-rocm`, then invokes only the
established softmax generator. The canonical gfx1151 package retains typed
Tile and Target IR, produces an ELF HSACO plus an ordered f16/f32 launch
descriptor, and registers an exact `rocm_gfx1151` descriptor consumer. The
LLVM 23/ROCm 7.14 WSL host compiled a 5,808-byte f32 image and eight f16/f32
descriptor routes matched the established oracle across boundary `(1,1)`,
aligned `(4,256)`, ragged `(3,17)`, and multi-stride `(2,257)` shapes on the
visible gfx1151. Invalid dtype, dynamic-shape, result, binding, shape-guard,
and scalar contracts reject before submission. The package now asks AMD clang
for the matching `--rocm-path` selection and records SHA-256 identities for
OCML, OCKL, and the five selected OCLC math/wave/ISA/ABI controls; those
identities participate in cache and toolchain fingerprints without retaining
installation paths. Cold and warm packages retain identical image, payload,
and library identity. The incumbent `rocm_softmax_compiled` executor and
production selector remain unchanged.

The isolated serial comparison supplies nine paired trials per row
with HIP-event and allocation/copy-inclusive timing kept separate. All eight
f16/f32 rows are correct; both code objects use 16 VGPR, 14 SGPR, 32 bytes LDS,
and no private segment or spills. After fixing a retained-route HIP-module
cleanup leak discovered by the comparison, the first run isolated the two fp16
end-to-end misses to repeated deterministic artifact/image/descriptor hashing
(about 107 us combined), not device work. Frozen identities are now memoized
while descriptor validation remains per-launch. The unchanged gate passes all
eight rows: device speedups span 0.981--1.008x and allocation/copy-inclusive
end-to-end speedups span 0.979--1.022x. ROCM-E2E-1 is closed with no selector
change; runtime text-route retirement remains an explicit follow-on decision.

**ROCM-E2E-2 is closed (2026-07-19).** Its reduction slices map the
existing `tile.reduce_kernel(X, O, Outer, AxisExtent, Inner)` carrier to typed
`tessera_rocm.reduce` Target IR and an architecture-owned 256-thread
workgroup-per-output kernel. The legacy four-argument row-reduction directive
remains ABI-compatible; only the typed carrier selects the five-argument form.
Canonical packaging covers f16/bf16/f32 input with f32 output. Exact gfx1151
sum/mean/max rows pass axis 0, middle-axis keepdims, and last-axis oracles.
Address hoisting plus a last-axis specialization closes all nine applicable
comparison gates: end-to-end speedups span 0.934--1.020x and layout-equivalent
last-axis device speedups span 0.935--1.011x. Axis-0/middle device values remain
diagnostic because the retained route's host transpose is outside its event
interval; the typed route directly consumes the strided layout. Host delta
spans -0.041 to +0.155 ms. The selector and runtime-authored route remain
unchanged.

The next breadth slice maps shared `tile.paged_kv_read_kernel` to typed
`tessera_rocm.paged_kv_read`, a direct f32/i32 gather HSACO, and a guarded
three-buffer/seven-scalar descriptor. Runtime validation rejects invalid page
indices before launch, and four exact gfx1151 ranges cover single-token,
page-crossing, ragged, and full-capacity permuted-page reads bit-for-bit. The
third breadth slice maps shared `tile.moe_dispatch_kernel` to typed
`tessera_rocm.moe_dispatch` and a guarded f32/i32 direct-gather descriptor.
Three exact-device shapes, including H=257, match indexed NumPy gathers
bit-for-bit. The paired movement record retains the production routes: typed
paged-KV is 0.960x device but 0.282x end-to-end, while typed MoE dispatch is
0.826x end-to-end. These are measured non-winners, not reasons to weaken the
10% threshold. Reduction plus the two scoped movement families therefore have
complete artifact, launch, numerical, negative, and route-disposition evidence;
future family migrations require separately scoped work.

ROCm typed-spine expansion is additionally gated by the gfx1151 datatype
totality contract. It distinguishes native scalar/vector formats, WMMA inputs,
accumulator-only formats, planned storage, and architectural negatives across
every canonical and planned dtype. In particular, RDNA3.5 scalar FP64 support
does not imply FP64 WMMA or current Tessera target registration; IU4 hardware
does not imply first-class packed-int4 storage; and FP8/BF8 remains a named
gfx1151 negative.

**NVIDIA-E2E-2 is closed for the available SM120 host (2026-07-20).** The first slice replaces the shared
SM90 alias behavior with distinct SM90/SM100/SM120 front and target pipelines.
SM90 alone selects the proven WGMMA/Hopper marker lowering; SM100 and consumer
Blackwell SM120 retain target-tagged typed matrix carriers. Straight-line
Graph→Tile copies now mint `!tile.async_token`, waits retire the exact tokens,
and matrix consumers retain the dependency through TMA lowering. Direct
FileCheck proof covers all three builders, but SM90 and SM100 native execution
remain explicitly exact-device gated. That early breadth statement is now
superseded by the expanded SM120 record below. Remaining NVIDIA-E2E-2 work is
limited to measured route dispositions; unavailable multi-GPU, SM90, and SM100
proof has an explicit hardware-deferred terminal instead of an ambiguous open
state.

The exact-architecture evidence boundary is explicit:

| Target | Level-C state | Reason |
|---|---|---|
| `nvidia_sm90` | deferred / exact-device unavailable | The SM120 host cannot validate Hopper WGMMA/TMA execution, resources, or timing. Compile-only artifacts remain Level B; a Hopper host must reopen a separate follow-up. |
| `nvidia_sm100` | deferred / exact-device unavailable | Consumer Blackwell has no tcgen05/TMEM and cannot close datacenter Blackwell execution. Compile-only artifacts remain Level B; an SM100 host must reopen a separate follow-up. |
| `nvidia_sm120` | ready / Level C for recorded families | Canonical matmul, f16/f32 softmax and arbitrary-axis reduction, f16/bf16/TF32/FP8 epilogue, forward/backward attention, paged-KV, ReplaySSM, and f16/bf16/f32 local MoE descriptors execute on the RTX 5070 Ti. Exact multi-device transport is a formal hardware-deferred terminal. |

The family migration order preserves one typed producer and one launch ABI per
semantic family rather than routing marker operations through a generic CUDA
kernel:

| Family | SM120 canonical state | Next proof boundary |
|---|---|---|
| matmul | f64, TF32, bf16, f16, FP8, FP6, MXFP4, INT8, NVFP4 ready on SM120 | 19/20 higher-amortization rows strict-stable; bimodal TF32 row is terminal non-promoting; retain selectors |
| softmax | f16/f32 ready | four canonical/production rows strict-stable; cooperative production route wins both domains and is retained |
| reductions | f16/f32 arbitrary-axis sum/mean/max with keepdims and serial/cooperative-128 schedules ready on SM120 | stable two-domain corpus; production selector retained under the no-unregistered-promotion rule |
| fused epilogues | f16/bf16/TF32/FP8 E4M3/E5M2 bias × activation × residual matrix ready on SM120 | stable accepted corpus; no selector promotion without a registered material-benefit threshold |
| attention | f16/f32 forward MHA/GQA/MQA plus bias/window/softcap/deterministic dropout; deterministic f16/f32 backward replays the same counter mask with f32 accumulation and zero workspace | four forward comparison rows strict-stable and production wins both domains; atomic/split backward production evidence remains authoritative |
| paged-KV | f32 direct gather plus compiler-owned fused page-table/token-index/causal-offset attention descriptor; remap, boundary, malformed-table, direct/staged evidence ready on SM120 | timing-domain disagreement retains the selector; controlled native-Linux evidence is a future promotion prerequisite |
| ReplaySSM | resident state/ring context loads compiler-owned Tile→PTX decode/flush images; session workspace, transition, ordering, numerical proof, and all 10 two-domain timing rows are ready on SM120 | closed; retain existing route |
| MoE | canonical int32 metadata plus f16/bf16/f32 compiler-owned Tile→PTX dispatch/combine/ragged-grouped-GEMM execution with f32 accumulation | six local comparison rows strict-stable; timing domains do not agree across all routes, so retain existing selector |
| transport | local-device dispatch → expert compute → combine evidence plus explicit rank/device topology and one-process multi-device NCCL all-reduce/gather/reduce-scatter/all-to-all execution are implemented | exact two-or-more-GPU proof is hardware-deferred on the one-GPU SM120 host and must be reopened on a multi-GPU system |

The stateful image landing keeps host-owned session machinery where it belongs
while moving device code ownership across the canonical seam. ReplaySSM's
resident CUDA context loads compiler-produced decode/flush PTX and preserves
its allocations, ring slots, events, and stream ordering. Compiler-owned MoE
dispatch/combine/grouped-GEMM candidates execute over the canonical grouped
metadata. The final strict-stable comparison retains the production selector
because the timing domains do not agree across all three routes. An explicit
NCCL topology and executor now exist, but exact multi-rank Level-C evidence is
hardware-deferred because
the RTX 5070 Ti WSL host exposes only one device; a rejected two-device binding
is contract proof, not collective execution proof.

The paired remaining-dtype/reduction corpus now covers production-sized TF32
and FP8 fused epilogues and serial/cooperative f16/f32 reductions against the
existing production CUDA routes. First-use compilation/cache fill is outside
steady-state timing; every sample discards one launch and amortizes each timing
domain over the next ten. Across two disjoint time-interleaved 100-sample
cohorts, 29/30 candidates satisfy the strict WSL 4% rule. The sole 4.099%
fp16-mean production end-to-end row is accepted under the explicitly approved
4.15% WSL rounding margin and cannot promote a selector. Resources, spills,
image/resource fingerprints, and cold/warm or first/second-use cache state are
retained; no selector changes.

The final SM120 comparative corpus has 14/14 strict-stable rows under the WSL
4% rule. It compares canonical Tile descriptors against production CUDA for
softmax, forward attention, and local MoE dispatch/combine/grouped GEMM in both
timing domains, with first-launch discard, 100-launch end-to-end amortization,
per-candidate clock conditioning, cold/warm image state, and resource
fingerprints. Production softmax and attention win both domains; MoE lacks
cross-domain consensus across all three routes. All existing selectors are
therefore retained. ReplaySSM's higher-amortization record is also 10/10 stable.

The final dtype matrix uses 31 samples, 10,000 resident launches, and 50
allocation/copy-inclusive launches per sample. Nineteen of twenty rows are
strict-stable. TF32 `256x256x256` remains bimodal at 7.02% device-event delta
with stable end-to-end timing and is assigned an explicit non-promoting
terminal; it does not block SM120 semantic closure or authorize a selector
change.

The second landing slice moves static f16/f32 last-axis softmax through the same
canonical seam. A registered `tile.softmax_kernel` carries X/O/Rows/K plus
explicit f16/f32 storage, f32 accumulation, and axis semantics; the SM120
backend extends f16 loads to f32, emits stable max-shifted loops and typed
`nvvm.ex2`, and truncates only at the f16 output boundary. LLVM 23 produces
`sm_120a` PTX and the storage-keyed descriptor launches through the shipped
CUDA-driver bridge. Exact RTX 5070 Ti rows cover both storage types,
rank-2/rank-3 flattening, `K=16/48/64/300`, extreme logits, invalid output shape
rejection, cold/warm identity, and ptxas resource fields. The correctness-first
thread-per-row candidate does not replace the existing cooperative CUDA
softmax selector. The final four-row comparison is strict-stable in both timing
domains and retains that route because it wins both domains.
The operation now carries backend-neutral `exp_mode="approx_exp2"` and an
explicit `ftz=false` bit. The SM120 lowering maps that choice to
`ex2.approx.f32`; the CUDA math contract records PTX's full-range 2-ULP bound
and keeps this route distinct from both IEEE arithmetic and function-specific
CUDA libdevice calls. The contract version participates in native cache
identity, so changing FTZ, approximation, or rounding semantics cannot reuse a
stale image merely because the optimization level stayed at `-O3`.
CUDA Math API totality now also has a compile-only SM120 inventory for scalar
integer math, bit operations, packed integer dot products, numeric/bit casts,
and 2x16/4x8 SIMD. The shared rounding registry adds the missing toward-positive
and toward-negative modes and maps the exact CUDA cast suffix set
RN/RD/RU/RZ; nearest-away and stochastic modes fail closed for CUDA casts. The
inventory deliberately leaves every new Target-IR and runtime state `planned`:
successful `nvcc -arch=sm_120a` header compilation is not Level-B or Level-C
proof.
The PTX 9.3 audit adds a fourth, lower layer beneath CUDA language types. Every
SM120 dtype row now names its physical PTX storage register, fundamental versus
alternate/sub-byte format class, and Tensor Core operand register. BF16 is
corrected from scalar/vector `native` to `conversion_only`: PTX stores scalar
BF16 in `.b16`, while packed BF16 matrix operands use `.b32`. TF32 similarly
remains fp32 storage but is an alternate instruction format carried in `.b32`;
FP8/FP6/FP4 formats use bit-size registers and never become fundamental PTX
types. A companion memory contract rejects packed/vector accesses as one atomic
unit, mixed-size-race guarantees, reduction-as-acquire, texture coherence, and
the inference that ordered host submission creates intra-kernel ordering.
Direct `ptxas -arch=sm_120a` fixtures assemble the fundamental register set and
prove that `bf16`, `tf32`, `e4m3`, and `u8x4` declarations reject as
non-fundamental formats.

The dtype-totality slice adds one SM120 contract across canonical storage,
math mode, scalar/vector CUDA types, Tensor Core formats, compiler fragments,
and runtime readiness. CUDA 13.3 compiles the fp64/fp32/fp16/bf16/FP8/FP6/FP4
scalar-vector surface. TF32 is guarded as an fp32-only math mode. Tensor Core
Target IR/PTX covers TF32/bf16/fp16/FP8/FP6/FP4/int8. The canonical descriptor
lane now executes general-shape BF16, explicit fp32-storage TF32 math, FP8
E4M3/E5M2, and INT8 with int32 accumulation, in addition to f16/NVFP4. FP64
m8n8k4 DMMA now has its distinct Tile lane map, f64 descriptor/bridge ABI,
masked ragged materialization, and aligned/ragged RTX 5070 Ti numerical proof.
FP6 E2M3/E3M2 use the
ptxas-validated m16n8k32 `kind::mxf8f6f4` UE8M0/1X contract; OCP/MXFP4 uses
m16n8k64 `kind::mxf4` UE8M0/2X. Their packed-memory Tile materializers, runtime
ABIs, numerical execution, and exact-device comparisons are complete. The
checked-in dtype-matrix recorder retains cold/warm image identity, ptxas
resources, CUDA-event batches, allocation/copy-inclusive timing, and disjoint
sample-interleaved two-run stability without promoting a selector. This slice
also removes a
wrong OCP-FP4→NVFP4 alias and requires the shared MMA selector to distinguish
MXFP4 UE8M0 scales from NVFP4 UE4M3 scales. It changes no production selector.

Device libraries are part of the LLVM-stage spine contract. The native-image
schema now records each linked library by logical name, SHA-256 content digest,
and link mode; absolute installation paths are deliberately excluded. NVIDIA's
direct LLVM→NVPTX path discovers CUDA `libdevice.10.bc`, fingerprints it as a
toolchain input, and runs `llvm-link --only-needed` only when translated LLVM
IR retains `__nv_*` declarations. ROCm must populate the same record from the
matching compiler-driver-selected OCML/OCKL/OCLC set: the ROCm driver and
`--rocm-path` remain authoritative because the selected control bitcode must
match target flags. Device-library identity therefore invalidates stale caches
without imposing one backend's physical library set on another.

The next software-actionable backend item is **APPLE-E2E-1**. Any future SM90,
SM100, native-Linux SM120 promotion, or multi-GPU NVIDIA evidence must reopen a
separate exact-hardware follow-up rather than changing this closed record.

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
