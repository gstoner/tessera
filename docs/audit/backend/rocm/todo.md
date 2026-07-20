---
last_updated: 2026-07-19
audit_role: plan
plan_state: open
scope: ROCm backend implementation and exact-device proof
---

# ROCm backend TODO

Cross-backend sync `STATEFUL-TRANSPORT-FOUNDATION-2026-07-19`: the shared launch
workspace schema now distinguishes per-launch scratch from session-persistent,
preserved state. ReplaySSM and MoE metadata contracts are portable, but this
NVIDIA slice changes no HIP allocation, wave schedule, event/ring protocol,
resource claim, timing row, or selector. ROCm's proven gfx1151 resident handle
must map its lifecycle to the shared descriptor in a ROCm-owned follow-up;
CUDA local-device bandwidth supplies no gfx1151 or multi-rank evidence.

Cross-backend sync `NVIDIA-E2E2-STATEFUL-REDUCE-2026-07-19` extends the shared
Tile surface with explicit ReplaySSM decode/flush, MoE dispatch/combine/grouped
GEMM, and `Outer/AxisExtent/Inner` reduction carriers, plus a backend-neutral
rank/device topology fingerprint. ROCM-E2E-2 must assess mapping these carriers
to the existing HIP generators and RCCL execution; this is follow-up required,
not CUDA parity evidence. ROCm inherits no warp schedule, PTX ABI, NCCL call,
resources, timing, or selector, and gfx1151 remains unsupported for FP8 WMMA.

This is the working ROCm implementation queue. It consolidates the open actions
from [`ROCM_AUDIT.md`](ROCM_AUDIT.md), the portable Tile fragment work in
[`tile_fragment_abi.md`](../../../architecture/proposals/tile_fragment_abi.md),
the serving work in
[`REPLAYSSM_PLAN.md`](../../roadmap/REPLAYSSM_PLAN.md), and the generated exact-
target status in [`rocm_target_map.md`](../../generated/rocm_target_map.md).

The generated target map and runtime/conformance dashboards remain the status
authorities. This file owns sequencing and completion gates; it must not promote
an artifact-only row by prose.

## Rules of completion

A ROCm item is complete only when all applicable evidence refers to the same
exact target:

1. the compiler emits a target-valid artifact;
2. the artifact assembles for the named gfx architecture;
3. the production runtime launches it;
4. device output matches a numerical oracle, including required ragged cases;
5. performance work compares against the retained production path with device
   timing and an explicit promotion threshold;
6. evidence records the actual `evidence_arch` and updates the generated target,
   runtime, and conformance views.

The generic `rocm` target is a family rollup. Evidence from gfx1151 must never
promote gfx1200, gfx1201, gfx1250, gfx942, or gfx950.

## Current baseline—not TODO

The following foundations already execute on `rocm_gfx1151` and should be
preserved while completing this queue:

- general f16/bf16/int8/int4 WMMA GEMM with tiled K loops, ragged boundaries,
  fused epilogues, runtime launch, and size-aware macro-tiles;
- compiler-generated flash-attention forward/backward, GQA/MQA/MHA, sliding
  windows, bias, logit soft-capping, and causal/ragged handling;
- linear, sparse, recurrent, DeltaNet, normalization, activation, reduction,
  positional, MoE, grouped-GEMM, and selective-SSM lanes listed as verified in
  the generated exact-target map;
- runnable global-to-LDS asynchronous copy and structured layout consumption;
- a versioned PLHD paged-KV ABI with an i32 logical-to-physical page table;
- cooperative sparse attention and resident top-k with committed comparative
  ratchets.
- Cross-backend sync `NVIDIA-TEST5-2026-07-16`: the shared autotune corpus v3
  adds compiler/resource, cold/warm, cache, and two-run stability evidence while
  retaining v1/v2 loading. ROCm corpus round-trip and warm-start behavior are
  parity validated by `test_rocm_measured_autotune.py`; no NVIDIA schedule,
  resource claim, or selector decision applies to gfx1151 or other AMD targets.
- Cross-backend sync `LLVM23-NVIDIA-2026-07-16`: ROCm's lit configuration uses
  LLVM 23's supported internal shell, matching its already recorded 32/32
  gfx1151 WSL proof. The Ubuntu bootstrap now probes the apt.llvm.org suite so
  current LLVM 23 snapshot packages work on Resolute as well as the documented
  Noble host. No NVIDIA lowering, route, timing, or resource evidence is
  transferred to ROCm; no new AMD exact-device claim is made here.
- Cross-backend sync `NVFP4-TILE-SCALES-2026-07-16`: shared typed Tile IR now
  permits logical `scale_a`/`scale_b` fragments only on NVFP4 MMA descriptors.
  This is not applicable to enabled gfx1151 WMMA matrix forms: gfx1151 has no
  NVFP4 block-scaled matrix instruction. ROCm retains its named unsupported
  capability result; NVIDIA nibble packing and scale-selector lane maps are not
  transferred.
- Cross-backend sync `PR420-REVIEW-2026-07-17`: the NVFP4 scale-origin repair
  and canonical `fp16` selector alias are NVIDIA-only and do not change ROCm
  fragment layouts, dtype support, runtime ABI, or exact-device evidence. The
  shared Ubuntu bootstrap now installs `ca-certificates`, `wget`, and `gnupg`
  before probing apt.llvm.org and removes only its version-owned stale source
  file before the prerequisite update. This is parity validated as setup
  infrastructure; it does not transfer CUDA schedules or make a new gfx claim.

## LLVM/MLIR 23 and ROCm 7.14 transition evidence

**Status: host build and gfx1151 correctness ratchets complete in WSL
(2026-07-16); bare-metal-only gates remain open.**

The project-wide compiler floor is now a matched LLVM/MLIR 23 toolchain. On the
gfx1151 WSL host, the validated configuration uses upstream Ubuntu LLVM/MLIR
23.0.0 for Tessera's C++ compiler and TheRock Core SDK 7.14 for HIP, HIPRTC,
device libraries, and the HIP compiler. Mixing the former LLVM/MLIR 23 build
with TheRock's LLVM 23 `ocml.bc` was rejected after the reader reported an
LLVM-bitcode attribute-version mismatch.

The clean `build-rocm-7.14-llvm23-clean` configuration and full Ninja build
pass.
The migration includes MLIR 23's removed dialect property switch, Queue
TableGen name collision, greedy-rewrite API split, tiling-interface alignment
overload, vector multi-reduction API split, and MFMA control operands becoming
attributes. Validation on the visible `gfx1151` device records:

- ROCm Target IR lit: **32/32 pass**;
- compiled ROCm correctness corpus on gfx1151: **1280/1280 pass**;
- valid baseline/performance ratchets: **21/21 pass**;
- combined paged-KV, ReplaySSM, portable Tile, grouped GEMM/SwiGLU, and
  architecture sweep: **86/90**, with only four source-confirmed invalid
  zero-event assertions remaining and no gfx1250/gfx1251 LLVM 23 failures;
- HIP version **7.14.60850**, TheRock HIP clang **23.0.0git**, and upstream
  LLVM/MLIR **23.0.0**.

This is WSL exact-gfx1151 correctness evidence, not bare-metal transport
evidence and not evidence for any sibling architecture.

## Status ledger

| ID | State | Current outcome |
|---|---|---|
| LLVM23/ROCm 7.14 | complete on gfx1151 WSL | Clean build, 36/36 ROCm lit, 1280/1280 compiled correctness, and 21/21 valid performance ratchets pass; the combined sweep is 86/90 with four zero-event-only failures. |
| ROCM-TILE-1 | complete on gfx1151 | Portable f16/bf16/int8/int4 fragments execute and compare on gfx1151. Other architectures are owned by ROCM-1 through ROCM-5. |
| ROCM-9 | complete on gfx1151 | Non-identity paged-KV direct and gather routes execute, compare, and have a measured serving decision. |
| ROCM-REPLAY-1 | complete on gfx1151 | Persistent state, flush/rollback, block submission, asynchronous ring, lifetime proof, and the wider performance matrix are committed. |
| ROCM-6 | open revalidation; timing blocked | LLVM/MLIR 23 + ROCm 7.14 correctness is green for G6-A/B/C, but WSL HIP events return invalid zero durations. Existing production choices stay in force pending valid paired device timing. |
| ROCM-8 | blocked | Bare-metal gfx1151 access is required; WSL characterization cannot close it. |
| ROCM-1/2/3 | open, access-gated | P0 exact-device execution on gfx950, gfx1201, and gfx1250 is the active release frontier. |
| ROCM-4a/4b | open, access-gated | P1 compatibility execution on gfx1200 and gfx942 follows the P0 packet. |
| ROCM-5 | landing, exact-device closure open | Architecture-owned descriptors and cross-assembly exist; numerical and performance closure depends on ROCM-1 through ROCM-4b. |
| ROCM-E2E-1 | complete on gfx1151 | The f16/f32 pilot lowers typed Tile IR to `tessera_rocm.softmax`, packages an ELF HSACO and descriptor, executes across the exact-device boundary/aligned/ragged matrix, rejects invalid contracts, retains driver-selected device-library plus cold/warm identity, and passes isolated device and end-to-end non-regression. |
| ROCM-E2E-2 | complete on gfx1151 | Reduction covers f16/bf16/f32 input with f32 output and passes all paired gates. Direct f32/i32 paged-KV and MoE dispatch have typed artifact/descriptor, negative, exact-device, and retained-route evidence; both movement descriptors are measured non-winners, so production routes remain selected. |
| ROCM-DTYPE-1 | landing on gfx1151 | The RDNA3.5 contract now totals every canonical and planned dtype across scalar/vector, WMMA input, accumulator, and Tessera readiness roles with archived-opcode evidence; registration gaps remain explicit. |

## Recommended open-work order

Completed and measured-non-winning gfx1151 work is intentionally absent from
this queue. The local WSL host may prepare artifacts and harnesses, but only the
named exact device can satisfy an execution gate.

| Order | ID | Work | Access state | Completion gate |
|---:|---|---|---|---|
| 1 | ROCM-2 | Run the common P0 packet on Radeon AI PRO R9700 `gfx1201` | owner and reservation required | RDNA 4 WMMA-v2 f16/bf16 plus enabled FP8/integer forms assemble, launch, match aligned/ragged oracles, and record resources and timing. |
| 2 | ROCM-1 | Run the common P0 packet on MI350-series `gfx950` | owner and reservation required | CDNA 4 matmul, flash attention, softmax, and GELU launch and compare; low-precision breadth advances only with physical-layout proof. |
| 3 | ROCM-3 | Run the common P0 packet on MI455X `gfx1250` | owner and reservation required | The upstream-LLVM artifact joins to a launch/numerical proof; WMMA-v2 properties and fragment layout match the device. |
| 5 | ROCM-6 | Revalidate G6-A/B/C with valid paired device timing | bare-metal gfx1151 or repaired event timing required | Original correctness, resource, aligned/ragged, dtype, device-time, and E2E gates are rerun under LLVM/MLIR 23 + ROCm 7.14 before reaffirming or changing production. |
| 6 | ROCM-8 | Measure copy versus mapped-host memory on bare-metal `gfx1151` | bare-metal owner and reservation required | Repeated kernel-only and end-to-end measurements establish a stable crossover without using WSL evidence. |
| 7 | ROCM-4b | Retain compatibility proof on MI300X/MI325X `gfx942` | owner and reservation required | f16/bf16 MFMA plus retained matmul/attention/softmax/GELU paths launch and compare. |
| 8 | ROCM-4a | Add Radeon RX 9000 `gfx1200` exact-device proof | owner and reservation required | Matmul launches and compares; unsupported forms reject stably. |
| 9 | ROCM-5 | Close the architecture-owned fragment umbrella | depends on ROCM-1 through ROCM-4b | Every enabled family/dtype has exact-device packing, numerical, resource, and timing evidence, or an explicit unsupported/deferred state. |
| 10 | ROCM-TEST-1 | Validate ROCm host-free compiler ownership | available on the ROCm build host | Build the declared ROCm compiler capability set, explicitly select or exclude foreign-backend compiler tests, and retain command, build flags, tool path, node IDs, and diagnostics; no CUDA/Apple-only build assumption blocks the ROCm lane. |

## ROCM-E2E-1: typed softmax compilation spine

**Status: complete on `gfx1151` (2026-07-19).**

The first slice consumes the shared `tile.softmax_kernel(X, O, Rows, K)`
semantic envelope and emits `tessera_rocm.softmax` from the architecture-owned
`lower-tile-to-rocm` pass. The canonical ROCm pipeline invokes the existing
softmax generator after that adaptation; no other standalone ROCm generator
was appended. Python supplies the typed Tile request and packages the resulting
Target IR, ROCDL-produced ELF HSACO, ordered f16/f32 ABI, shape guards, gfx1151
workgroup policy, and launch descriptor. The generic runtime validates that
descriptor before the exact `rocm_gfx1151` hook allocates, submits, copies back,
and cleans up the HIP resources.

Host WSL validation used LLVM/MLIR 23 with the ROCm 7.14 build and a visible
Radeon 8060S `gfx1151`. The focused lit fixture passed, the Python/registry
slice passed 154 tests, real packaging produced a 5,808-byte ELF HSACO, and the
expanded focused slice passed 24 tests. Eight exact descriptor launches cover
f16/f32 at boundary `(1,1)`, aligned `(4,256)`, ragged `(3,17)`, and
multi-stride `(2,257)` shapes and match the stable-softmax oracle. Host-free
negatives reject unsupported dtype, dynamic shape, mismatched result, runtime
dtype/shape binding, and scalar contracts before submission.

AMD clang is authoritative for the `gfx1151` `--rocm-path` selection. The
native image records content digests for OCML, OCKL,
`oclc_unsafe_math_off`, `oclc_finite_only_off`,
`oclc_wavefrontsize64_off`, `oclc_isa_version_1151`, and
`oclc_abi_version_600` with `compiler_driver` link mode and no installation
paths. Those records enter both cache and toolchain identity. An exact cold then
warm package retained identical image, payload, and library identities. The
legacy runtime-authored `rocm_softmax_compiled` path remains intact and no
selector changed.

The isolated serial performance gate is recorded in
`benchmarks/baselines/rocm_gfx1151_e2e_softmax_comparison.json`. Nine
alternating paired trials per row keep 100-launch HIP-event timing separate
from allocation/copy-inclusive `runtime.launch` wall time. All eight f16/f32
aligned, ragged, and multi-stride rows remain numerically correct with exact
route parity. Compiler/retained resources match at 16 VGPR, 14 SGPR, 32 bytes
LDS, zero private segment, and zero spills. Device speedups span
0.981--1.008x and pass the 10% non-regression gate on every row.

The first A/B run exposed that the retained `rocm_softmax_compiled` executor
freed device buffers but did not unload its HIP module and leaked resources on
several failures. That lifecycle defect is fixed before accepting comparison
evidence. With both routes cleanup-complete, the first A/B run isolated the two
fp16 misses to repeated deterministic identity work on the descriptor route:
`RuntimeArtifact.artifact_hash` cost about 84.8 us, the image and descriptor
digests about 22 us together, and required per-launch validation about 24 us.
The immutable artifact, image, and descriptor identities are now memoized;
contract validation still runs on every launch. The unchanged serial gate then
passes all eight rows: device speedups span 0.981--1.008x and end-to-end
speedups span 0.979--1.022x, including fp16 `(128,256)` at 1.009x and
`(64,1024)` at 0.997x. Resources and numerical results remain identical.
ROCM-E2E-1 is complete. The incumbent route and production selector remain
unchanged; retiring runtime text synthesis is an explicit ROCM-E2E-2 route
decision, not an implication of this measurement.

Cross-backend sync `E2E-FROZEN-IDENTITY-CACHE-2026-07-19`: deterministic hashes
for frozen runtime artifacts, native images, and launch descriptors are cached
without changing their serialized values or validation rules. CUDA and Metal
schema parity is validated; no sibling ABI, schedule, runtime route, timing
claim, or selector changes.

## ROCM-E2E-2: typed directive and generator breadth

**Status: complete on `gfx1151` (2026-07-19).**

The reduction breadth slices consume the already-shared
`tile.reduce_kernel(X, O, Outer, AxisExtent, Inner)` carrier. ROCm lowering
requires explicit f16/bf16/f32 storage, f32 accumulation/output, normalized
axis, keepdims, sum/mean/max semantics, NaN propagation, and the portable
serial schedule, then selects its own 256-thread workgroup-per-output
implementation. The existing legacy four-argument row-reduction directive
remains valid; only the typed carrier selects the five-argument
`outer_axis_inner` ABI.

Canonical packaging retains Tile IR and typed `tessera_rocm.reduce` Target IR,
builds an ELF HSACO, records the driver-selected device-library identities, and
emits shape guards plus `Outer/AxisExtent/Inner` scalars. The exact gfx1151 WSL
host passes f16 sum on axis 0, bf16 mean on a middle axis with `keepdims`, and
f32 max on the last axis against NumPy. The shared Tile verifier now admits
bf16 reduction storage; NVIDIA keeps its backend-specific f16/f32 boundary and
Apple mappings are unchanged.

The isolated serial comparison is recorded in
`benchmarks/baselines/rocm_gfx1151_e2e_reduce_comparison.json`. Nine alternating
paired trials per row separate resident HIP-event timing from full
`runtime.launch` wall time across f16/bf16/f32 sum/mean/max and axis-0/middle/
last layouts. The kernel hoists arbitrary-axis base/stride calculations and
specializes the last-axis case. All nine comparison rows pass: end-to-end
speedups span 0.934--1.020x and the layout-equivalent last-axis device rows span
0.935--1.011x. Device-event values for axis-0/middle rows remain diagnostic,
not a promotion gate, because the incumbent performs an untimed host transpose
before launching a contiguous row kernel while the typed route directly reads
the original strided layout. Descriptor-minus-retained host overhead spans
-0.041 through +0.155 ms. The selector and retained route remain unchanged.

The next family consumes shared
`tile.paged_kv_read_kernel(Pages, Table, O, P, LP, PageSize, H, D, Start,
Tokens)`, lowers to typed `tessera_rocm.paged_kv_read`, emits a direct 256-thread
f32 gather with an i32 page table, and packages the three-buffer/seven-scalar
descriptor. The runtime validates shapes, range, dtypes, and every physical
page index before submission. Exact gfx1151 execution matches a non-identity
permuted-page oracle bit-for-bit at single-token, page-crossing, short-ragged,
and full-capacity ranges. Static contract and pre-HIP negatives cover bounds,
result shape, table dtype, and invalid physical pages. This adds no selector
promotion and does not replace the existing ROCM-9 HIP movement route.

The third family consumes shared `tile.moe_dispatch_kernel(X, Token, O, T, S,
H)`, lowers to typed `tessera_rocm.moe_dispatch`, and generates a direct
256-thread f32 gather from an i32 token-of-slot vector. Its three-buffer/
three-scalar descriptor rejects out-of-range indices before HIP. Exact gfx1151
execution is bit-exact at tiny `(1,1,1)`, ragged `(7,9,13)`, and wide
`(17,5,257)` shapes.

The final paired movement record is
`benchmarks/baselines/rocm_gfx1151_e2e_movement_comparison.json`. Typed
paged-KV is device-competitive at 0.960x but only 0.282x end-to-end versus the
retained HIP route; typed MoE dispatch is 0.826x end-to-end versus the retained
row gather. Both remain numerically exact. These measured non-winning results
close the route disposition without weakening the 10% promotion threshold:
ROCM-9 paged movement and the retained MoE transport stay selected, and no
runtime-authored route is retired. Future attention, backward, ReplaySSM, or
additional transport carriers are separately scoped breadth work rather than
silently extending ROCM-E2E-2. The item is complete for its reduction plus two
movement-family scope.

Cross-backend sync `ROCM-E2E2-REDUCE-2026-07-19`: this slice consumes the
existing shared reduction carrier and widens its storage verifier to bf16; the
scalar schema and public op registry are unchanged. NVIDIA retains an explicit
backend f16/f32 boundary, and Apple has no mapping change. The ROCm lowering,
five-argument ABI, HSACO, measurements, and selector state transfer no sibling
backend claim.

Cross-backend sync `ROCM-E2E2-PAGED-KV-2026-07-19`: ROCm consumes the existing
shared paged-KV carrier and public operation without changing either verifier
or schema. The new ROCm target directive, gather schedule, HSACO ABI, runtime
validation, and exact-device evidence are architecture-owned. NVIDIA's PTX
mapping remains parity validated at the carrier boundary; Apple retains its
independent Metal/MPS paged-cache routes.

Cross-backend sync `ROCM-E2E2-MOE-DISPATCH-2026-07-19`: ROCm consumes the
existing shared MoE dispatch carrier and public operation without changing
their verifier, dtype registry, or scalar ABI. The ROCm directive, direct
gather schedule, HSACO descriptor, pre-launch index validation, and gfx1151
evidence are architecture-owned. NVIDIA and Apple retain their independently
scheduled transport routes; no AMD timing, readiness, or selector claim
transfers.

## ROCM-DTYPE-1: gfx1151 datatype totality

**Status: landing on `gfx1151` (2026-07-19).**

`python/tessera/compiler/rocm_dtype_contract.py` separates ISA support from
Tessera execution readiness. Every canonical and planned/gated dtype has one
row; every positive architecture claim names an opcode present in the
AMD-PDF-derived `rdna35/instructions.json`. The community RDNA3.5 Markdown is
retained as a human-readable cross-check, not substituted for the checked-in
JSON source and hash.

| Format group | RDNA3.5 scalar/vector role | gfx1151 WMMA role | ROCm/LLVM state | Tessera target state |
|---|---|---|---|---|
| fp64 | native vector arithmetic and conversion | unsupported | available, focused validation open | unregistered |
| fp32 | native | accumulator only | validated | ready |
| fp16 | native | input; fp16/fp32 accumulator | validated | ready |
| bf16 | packed dot/WMMA | input; bf16/fp32 accumulator | validated | ready |
| int8 / uint8 | packed dot | IU8 input; int32 accumulator | int8 validated; uint8 validation open | int8 ready; uint8 planned-gated |
| int4 / uint4 physical IU4 | packed dot | IU4 input; int32 accumulator | validated | canonical int4 planned-gated; no first-class uint4 spelling |
| int16 / uint16 | packed/native vector arithmetic | unsupported | available, focused validation open | int16 unregistered; uint16 planned-gated |
| int32 / uint32 | native vector arithmetic | int32 accumulator only | available, focused validation open | int32 unregistered; uint32 planned-gated |
| int64 / uint64 | expanded instruction sequences | unsupported | available, focused validation open | int64 unregistered; uint64 planned-gated |
| bool | compare/mask logic | unsupported | available, focused validation open | unregistered |
| FP8/BF8, FP6, FP4, MX formats, NVFP4 | unsupported | unsupported | unsupported | not applicable or planned-gated negative |
| complex formats | no native complex datatype | unsupported | no native datatype | planned-gated |

The existing `ROCmTargetProfile.dtype_set` and target capability remain the
registered executable storage set `{fp32, fp16, bf16, int8}` and are checked
for exact equality with the new contract. Hardware support therefore cannot
silently promote fp64, packed int4, unsigned, or accumulator-only types.

Closure work is deliberately split: assess fp64 and int16/int32/int64 per-op
Target IR/runtime registration; finish the first-class packed-int4 physical
storage and signedness contract; retain named FP8/BF8 and smaller-float
rejection on gfx1151. No shared dtype spelling, backend selector, or production
state changes in this landing. Cross-backend sync
`ROCM-DTYPE-TOTALITY-2026-07-19` is ROCm-owned.

## ROCM-TILE-1: portable Tile fragments

**Status: complete on `gfx1151` (2026-07-14).**

The portable dialect defines `!tile.fragment`, `#tile.mma_desc`, `tile.view`,
`tile.fragment_pack`, `tile.mma`, `tile.fragment_unpack`, and `tile.store`.
ROCm now consumes the typed form for gfx1151 Wave32 f16, bf16, signed int8, and
signed int4 WMMA. The same logical fixture owns only descriptors and layouts;
the backend owns the physical VGPR fragments, packing, accumulator map, and
stores.

The checked-in proof is `gfx1151_tile_fragment_store.mlir` plus
`test_rocm_wmma_gemm_generated.py`. It covers parser/verifier materialization,
real ROCDL WMMA intrinsics, gfx1151 hsaco assembly, HIP module launch, and
exact-device comparison with aligned and ragged launch-level shapes. Negative
fixtures reject a contradictory B storage order and FP8/BF8 with named
diagnostics. The portable launch contract also reuses the production multi-tile
K-loop, fused bias/ReLU/GELU/SiLU, output conversion, and ragged-store generator
while preserving the portable operand ABI. The portable adapter feeds that
generator through an in-memory request rather than a temporary target-IR
directive, and per-column bias is loaded once per fast/edge output tile and
reused across its accumulator elements.

### Build steps

1. Resolve the portable MMA descriptor through ROCm's existing
   `MmaDescriptor` selector.
2. Implement pointer/layout-bearing A and B `tile.fragment_pack` for the
   gfx1151 Wave32 `16x16x16` WMMA lane.
3. Lower the accumulator fragment through `tessera_rocm.wmma` to the matching
   ROCDL/LLVM intrinsic.
4. Implement accumulator `fragment_unpack` and masked row-major store.
5. Reuse the existing epilogue contract for bias, ReLU/GELU/SiLU, output
   conversion, and ragged stores.
6. Add launch-level multi-tile grid and K-loop fixtures using the same logical
   Tile program as the NVIDIA path.

### Dtype order

| Target family | First contracts | Explicit guard |
|---|---|---|
| gfx1151 RDNA 3.5 | f16, then bf16, int8, int4 | FP8/BF8 WMMA must fail with a named capability diagnostic; gfx1151 does not have those matrix forms. |
| gfx1200/gfx1201 RDNA 4 | f16/bf16, then E4M3/E5M2 and integer forms supported by WMMA v2 | No promotion without matching RDNA 4 silicon. |
| gfx942/gfx950 CDNA | f16/bf16 MFMA first; add target-supported FP8/FP6/FP4 only from the CDNA descriptor table | Never route an RDNA WMMA fragment map into MFMA. |

### Required proof

- positive parser/verifier and negative layout/descriptor fixtures;
- structural ROCDL/LLVM intrinsic check;
- object/hsaco assembly for the exact target;
- aligned and ragged execute-and-compare through the production HIP bridge;
- a named rejection for every unsupported dtype/architecture pairing.

## ROCM-5: architecture-owned fragment layouts

**Status: compiler and exact-target assembly complete; remote exact-device
closure remains open (2026-07-15).**

The portable Tile program no longer inherits gfx1151's physical register map.
`rocm_fragment.py` and `ROCMFragmentLayout.h` select a data-only physical
descriptor after the exact gfx architecture is known. The descriptor owns the
matrix family, Wave32/Wave64 width, per-lane elements and registers, gfx11 input
replication, accumulator map, intrinsic ABI, and materialization readiness.
Python/C++ name-consistency tests and named family/dtype/shape errors prevent a
prefix fallback from silently selecting the wrong ABI.

The same logical pack/MMA/unpack/store fixture now lowers as follows:

| Architecture family | Physical contract | Enabled forms | Cross-assembly resources |
|---|---|---|---|
| gfx1100/gfx1151 | duplicated gfx11 Wave32 WMMA, padded accumulator map | f16/bf16/int8/int4 | f16: 25 VGPR, 6 SGPR |
| gfx1200/gfx1201 | dense SOA Wave32 RDNA 4 WMMA | f16/bf16/E4M3/E5M2/int8, K32 int4 | 18–35 VGPR, 8 SGPR |
| gfx1250/gfx1251 | K32 Wave32 WMMA-v2 with typed `modC` and reuse properties (`signA`/`signB` are not properties of the LLVM 23 f16/bf16 ops) | f16/bf16 | 28 VGPR, 6 SGPR |
| gfx90a | Wave64 CDNA2 MFMA | f16/bf16 | 12 VGPR, 12 SGPR |
| gfx940/gfx942 | Wave64 CDNA3 MFMA | f16/bf16 | gfx942: 14 VGPR, 14 SGPR |
| gfx950 | Wave64 CDNA4 MFMA | f16/bf16 | 14 VGPR, 14 SGPR |

All serialized rows use zero LDS and scratch and report zero VGPR/SGPR spills.
LLVM 23 still cannot serialize `gfx940` in the installed Ubuntu package;
gfx942 supplies the same-family object proof. The
repeated-median compiler/serializer harness is
`benchmark_rocm_arch_fragments.py`; the stable resource baseline is
`rocm_arch_fragment_resources.json`.

Exact-device execution remains deliberately narrower than cross-assembly. The
available gfx1151 host passes f16, bf16, signed int8, and signed int4 numerical
oracles. No RDNA 4, gfx125x, or CDNA performance or numerical claim is promoted
without matching silicon. The remaining ROCM-5 completion work is therefore:

1. run the shared fixture on gfx1200/gfx1201 and compare every enabled dtype;
2. run f16/bf16 on at least one gfx942 and one gfx950 device;
3. record kernel-only latency and measured occupancy on each exact device;
4. enable gfx125x FP8 and additional CDNA low-precision forms only after their
   physical packing map and numerical oracle are proven on matching hardware.

## ROCM-9: paged-KV serving

**Status: complete on `gfx1151` (2026-07-14).**

The ABI is fixed and portable:

- K/V pages: physical `[P, L, H, D]`;
- page table: i32 logical-page to physical-page mapping;
- token indices: i64 gather order;
- attention must not assume identity or contiguous page placement.

The production selector now mirrors the CUDA reference design while preserving
the portable ABI: it verifies both the retained gather→FA path and a direct
page-table consumer against one oracle, records HIP-event and full-call timing,
and warm-starts from D2. The exact-device fixture combines a non-identity table,
arbitrary token order, page crossings, causal decode offset, and MQA/GQA head
mapping. It also exposed and fixed a baseline bug: dense FA's query-zero causal
triangle is not the right-aligned `T-Q` decode mask, so gather→FA now supplies
that offset explicitly as additive bias.

### Completed work

1. `test_live_rocm_paged_gather_handles_permuted_pages` executes on gfx1151.
2. The combined direct fixture covers non-identity placement, arbitrary order,
   page crossings, MQA grouping, and a multi-query causal offset.
3. Gather→dense-FA remains the named baseline and shares the same oracle.
4. Direct HIP attention consumes PLHD K/V, the i32 page table, and i64 order in
   its K/V traversal without materializing dense K/V.
5. Both routes are HIP-event timed and full-call timed at 128, 512, 2,048, and
   8,192 cached tokens.
6. Both timing modes are committed to D2; full-call winners warm-start serving.

### Measured decision

Shape is `Q=1, Hq=Hkv=4, D=32, L=16`, causal decode, f32 PLHD storage. Times are
milliseconds on the exact `gfx1151` host. The direct candidate wins the current
host-pointer ABI end to end at every measured length, so it is the production
selection for these buckets. Gather→FA wins device-only time decisively; if the
cache ABI becomes device-resident, that evidence requires re-evaluating the
selection rather than carrying the host-pointer verdict forward.

| Cached tokens | gather→FA device | direct device | device winner | gather→FA E2E | direct E2E | serving winner |
|---:|---:|---:|---|---:|---:|---|
| 128 | 0.0232 | 0.1157 | gather→FA | 4.158 | 1.158 | direct |
| 512 | 0.0671 | 0.4510 | gather→FA | 4.350 | 2.389 | direct |
| 2,048 | 0.1721 | 1.7967 | gather→FA | 6.614 | 5.741 | direct |
| 8,192 | 0.6444 | 7.2107 | gather→FA | 22.674 | 19.466 | direct |

Evidence starts in:

- `tests/unit/test_paged_kv_rocm_abi.py`;
- `tests/unit/test_paged_kv_rocm_native.py`;
- `python/tessera/compiler/emit/rocm_hip.py`;
- `python/tessera/cache/paged_kv.py`;
- `benchmarks/rocm/record_paged_kv_corpus.py`;
- `benchmarks/baselines/rocm_gfx1151_paged_kv.json`;
- `benchmarks/baselines/autotune_corpus.json`.

## ROCM-REPLAY-1: ReplaySSM serving parity

**Status: complete on `gfx1151` (2026-07-14).**

The reference state ABI, flush policy, speculative rollback, and CUDA serving
implementation already define the semantics. The ROCm work must preserve those
semantics rather than introduce a backend-specific cache layout.

### Architecture plan

The implementation is a handle-side serving runtime, not a new Graph-IR state
type. `SSMStateHandle` remains the semantic authority for shape validation,
flush policy, checkpoint/restore, cloning, and the host reference mirror. A
scalar-A-only ROCm context is attached by `rocm_ssm_replay_state_handle`; if the
HIP toolchain or exact device is unavailable, the factory retains the same
handle and honestly falls back to the reference path.

The ROCm context owns these allocations for its entire lifetime:

- resident checkpoint `S0[B,D,N]` and scalar decay `A[D]`;
- fixed-capacity replay inputs `delta[L,B,D]`, `x[L,B,D]`, and `b[L,B,N]`;
- one scratch `c[B,N]` and `y[B,D]` for synchronous decode;
- an ordered ring of at least two asynchronous slots, each with pinned host
  staging for `(delta,x,b,c,y)`, device output `[L,B,D]`, and begin/completion
  HIP events.

One nonblocking producer HIP stream owns all device mutation. Append, block
submit, output-only reconstruction, and flush are ordered on this stream. The
output-only kernel reads resident `S0` plus replay inputs and writes only
`y[B,D]`. The flush kernel is the sole writer of `S0`; after it completes, both
host and device cursors return to zero. General `(D,N)` A continues through the
reference handle because the scalar-A factorization is the ReplaySSM contract
implemented by this first serving slice.

Rollback never launches a kernel: the host cursor rewinds and future appends
overwrite rejected positions. A speculative block must fit wholly before the
reserved flush boundary; mid-block flush is rejected. Synchronous block submit
and asynchronous submit use identical validation and token ordering. The host
mirror advances only after a successful enqueue, so failed submissions cannot
create a split-brain cursor.

An async slot has three states: free, leased to a result, and retired pending a
consumer event. Submission copies into pinned staging, enqueues H2D + ordered
decode kernels, retains device outputs, and records completion. `wait()` performs
the explicit D2H handoff and frees the lease. Device consumers receive an opaque
HIP buffer and producer-stream handle; `event.wait_on(stream)` establishes
cross-stream order, and `release(stream=...)` retires the lease only after that
consumer. Reusing a slot before its completion event is forbidden and reports
backpressure rather than silently synchronizing.

Correctness gates are layered:

1. source/ABI and shape guards run without a GPU;
2. output-only, flush, long decode, reset, rollback, speculative rejection, and
   block submission compare against `SSMStateHandle` on gfx1151;
3. multi-slot ordering, backpressure, wait/download, device lease, and release
   lifetime run on the live HIP runtime;
4. device-event and wall-time benchmarks compare replay against eager summary
   traffic and commit exact-architecture evidence.

### Completed work

1. The HIP context owns persistent S0 and fixed-capacity replay inputs.
2. Scalar-A output-only decode writes only `[B,D]`; shared history Gram scalars
   are computed once per `(token,batch)` rather than once per channel.
3. Only the flush kernel materializes and writes the full state.
4. Ordered block submission covers prefill and speculative verification.
5. The multi-slot ring uses pinned staging, a nonblocking producer stream,
   begin/completion events, backpressure, and opaque output leases.
6. Device consumers can wait on the producer event and retire a lease on their
   own HIP stream without a host download.
7. Summary, sequential output-only, and four-slot async modes have committed
   device/wall timing and analytical traffic rows.

### Required proof

- long decode across multiple flushes versus `SSMStateHandle`;
- rollback and speculative rejection equivalence;
- output-only and state-and-output route equivalence;
- ring ordering, backpressure, event waits, and device-buffer lifetime tests;
- representative serving rows committed to D2 with exact gfx1151 provenance.

All required proof cases execute in `test_ssm_rocm_replay.py`: 43-token decode
across repeated flushes, rollback and reset, speculative suffix rejection,
ordered block submission, two-slot backpressure, event timing, host wait, HIP
device-buffer exposure, cross-stream wait, and lease retirement. The broader
SSM/ReplaySSM regression sweep is 95 passed, 4 skipped.

### Measured decision

The table reports milliseconds per token for 64-token scalar-A decode on exact
gfx1151 (five repetitions). `summary` reconstructs output and writes resident
S0 every token. `output-only` is true sequential decode. `async` submits four
16-token blocks through the ordered slot ring.

| Shape `(B,D,N)` | Mode | Device ms/token | Wall ms/token | Speedup vs summary wall | State-traffic reduction |
|---|---|---:|---:|---:|---:|
| `1,64,64` | summary | 0.0770 | 0.1822 | 1.00× | 1.0× |
| `1,64,64` | output-only | 0.0400 | 0.1583 | 1.15× | 32.1× |
| `1,64,64` | async, chunk 16 | 0.0295 | 0.0337 | 5.41× | 32.1× |
| `1,128,128` | summary | 0.0769 | 0.1871 | 1.00× | 1.0× |
| `1,128,128` | output-only | 0.0410 | 0.1462 | 1.28× | 51.5× |
| `1,128,128` | async, chunk 16 | 0.0358 | 0.0403 | 4.64× | 51.5× |

#### Wider compiler matrix

The follow-up compiler matrix expands this to five geometries
(`1x32x16`, `1x64x64`, `1x128x64`, `1x128x128`, and batched `4x64x64`), token
lengths 16/64/256, capacities 16/64, and async schedules `(chunk=4,slots=2)`
and `(chunk=16,slots=4)`. It contains 75 exact-device rows, including forced
flush cases. Every row is checked against `SSMStateHandle`; maximum absolute
error is `5.07e-8`, and no ReplaySSM row loses to its matching summary row.

| Mode | Wall speedup min / median / max | Device speedup min / median / max |
|---|---|---|
| sequential output-only | 1.07× / 1.24× / 1.42× | 1.86× / 2.26× / 3.02× |
| async chunk 4, two slots | 3.11× / 3.69× / 5.06× | 1.96× / 2.58× / 3.66× |
| async chunk 16, four slots | 4.11× / 5.04× / 6.31× | 2.12× / 2.81× / 4.21× |

At `T=256, capacity=64` (three real flushes), chunk-16 throughput ranges from
22,998 tokens/s for `1x128x128` to 31,021 tokens/s for `1x32x16`. The narrowest
sequential win is 1.07× (`4x64x64`, `T=64`, `capacity=16`), so it is the first
candidate for a future performance ratchet rather than being hidden by the
matrix aggregate.

Evidence is in:

- `python/tessera/compiler/emit/rocm_hip.py`;
- `python/tessera/runtime.py`;
- `tests/unit/test_ssm_rocm_replay.py`;
- `tests/unit/test_ssm_rocm_replay_benchmark.py`;
- `benchmarks/rocm/benchmark_ssm_replay.py`;
- `benchmarks/baselines/rocm_gfx1151_ssm_replay.json`;
- `benchmarks/baselines/rocm_gfx1151_ssm_replay_matrix.json`.

## ROCM-6: performance redesign experiments

**Status: open revalidation under LLVM/MLIR 23 + ROCm 7.14; performance
blocked by invalid WSL HIP event timing (2026-07-16).**

The refreshed build passes the full required correctness matrices: G6-A passes
20/20 schedule rows over its four aligned/ragged/dtype cases; G6-B passes all
four cases with maximum difference `8.36e-6` versus one wave; G6-C passes all
six cases with maximum difference `3.13e-7` versus serial dK/dV. G6-B retains
its D=128 resource advantage: 121 VGPR with zero scratch/spills versus 218 VGPR
for one wave.

All three paired performance harnesses are blocked because ROCm 7.14 WSL HIP
event calls return success but report `0.0 ms`. The harnesses now reject
zero/non-finite samples and expose correctness-only mode. Until valid paired
device and E2E timing is collected, existing production choices stay in force
without claiming the old performance decisions were reaffirmed:

- G6-A remains non-production and is reopened for measurement;
- G6-B remains the current production route, with correctness/resources
  reaffirmed but performance revalidation open;
- G6-C remains non-production, with correctness reaffirmed but its prior
  performance rejection awaiting revalidation.

Evidence is recorded in
`benchmarks/baselines/rocm_gfx1151_rocm6_llvm23_rocm714_revalidation.json`.

### Phase 0: rebaseline older kernels with the current compiler

The 2026-07-14 exact-device survey reran the older compiler-generated GEMM and
flash-attention ladders with ROCm 7.2 and LLVM 23. This changes the premise of
G6-A but not G6-B or G6-C.

Generated f16 GEMM now reaches 12.23, 12.99, 23.51, and 26.29 TFLOP/s at
512³, 1024³, 2048³, and 4096³. The best tiles are respectively 2x4, 4x4, 2x4,
and 4x4. In particular, 4x4 is no longer a universal compiler-path VGPR cliff:
it wins at 1024³ and 4096³. At 4096³ the same 4x4 schedule reaches 25.90
TFLOP/s bf16, 29.53 TOP/s int8, and 31.32 TOP/s int4. G6-A must therefore begin
by rebuilding the size/dtype schedule ratchet with repeated medians and compiler
resource evidence; the two-wave/LDS reduction candidate is implemented only if
that renewed evidence still exposes an occupancy gap.

The older pipeline route is still specialized. It is 2.53x faster at 512³ but
only 1.002x, 1.007x, and 1.006x the direct route at 1024³, 2048³, and 4096³.
It must not become a general default.

Attention did not receive the same automatic compiler uplift. Forward remains
6.57--6.89 TFLOP/s at D=64 and 3.04 TFLOP/s at D=128; backward remains
3.92--4.15 TFLOP/s at D=64 and 2.13 TFLOP/s at D=128. Those numbers reproduce
the previous ceiling closely enough that the two-wave D=128 forward experiment
and split/reduced dK/dV backward remain the first kernel redesigns.

The survey is committed in
`benchmarks/baselines/rocm_gfx1151_legacy_compiler_rebaseline.json`. It is
exploratory evidence, not yet a promotion ratchet: the next measurement pass
must use repeated medians, preserve numerical oracles, and record code-object
VGPR/LDS occupancy before changing production selection.

That promotion pass is now complete. The 37-case, 185-row matrix uses nine
interleaved trials per tile so APU clock movement is paired rather than mistaken
for a schedule win. It covers square sizes 512 through 4096, transition sizes
1536/3072, model-shaped rectangular rows through `4096x11008x4096`, three
ragged rungs, f16/bf16/int8/int4, and bias/ReLU/GELU/SiLU epilogues. Every row
matches its numerical oracle and every tile is bitwise equal to the common
device result.

Assembler metadata explains why selection remains shape-dependent. Plain f16
1x1 and 2x2 use 51 and 136 VGPRs with no spills. The 2x4, 3x4, and 4x4 kernels
reach 256 VGPRs and report respectively 41, 257, and 392 spills with 108, 424,
and 736 bytes of scratch per work-item. Large aligned and wide shapes can
amortize that cost; small and ragged shapes cannot.

Production now uses the rows that clear both gates (at least 3% paired-median
gain and a win in at least 75% of interleaved rounds). Three near-ties retain
the previous selector: 3072-cube 4x4 over 3x4 (2.6%), the required large ragged
2x4 over 3x4 (2.1% after a 21-trial tie-breaker), and skinny-M=128 2x2 over
2x4 (2.8%). Evidence is committed in
`benchmarks/baselines/rocm_gfx1151_gemm_schedule_matrix.json`.

### G6-A: VGPR-bounded multi-wave GEMM

**Status: reopened for performance revalidation (2026-07-16); non-production
until the original gate passes.**

The ROCm 7.14 production correctness and performance ratchets pass, and the
existing repeated-median schedule baseline remains shape-dependent. A renewed
G6-A matrix was attempted at both required aligned f16 sizes plus the ragged
and int8 rungs. Under WSL, ROCm 7.14's HIP event API returned success but a
zero elapsed time for module-launch batches; the harness now rejects zero,
non-finite, and failed timing samples rather than emitting fabricated
throughput. Because the renewed measurement is invalid rather than negative,
it cannot reject or promote the two-wave/LDS-reduction design. Valid repeated
device timing must show whether the existing selector misses the stated 10%
opportunity before implementation proceeds.

- Split an output macro-tile across two Wave32 groups.
- Reduce bounded partial f32 accumulators through LDS.
- Keep per-wave accumulator pressure below the measured 4x4 VGPR cliff.
- Measure f16/bf16 at 2048³ and 4096³, ragged
  `2049x4093x2051`, and int8 at 2048³.
- Promote only with at least 10% median gain on both aligned f16 rungs, no rung
  slower by more than 3%, and all dtype/ragged oracles green.

### G6-B: two-wave online-softmax forward attention

**Status: current production route from ROCm 7.2 evidence; LLVM/MLIR 23 + ROCm
7.14 correctness/resources reaffirmed, performance revalidation open.**

- Give two waves one query tile and share K/V traversal.
- Merge per-wave online `(m,l,O)` state once per K/V tile.
- Measure `(1,8,512,64)`, `(1,8,1024,64)`, `(1,16,1024,128)`, and causal
  sequence 1009 at D=128.
- Promote only with at least 10% gain on both D=128 rungs and no D=64
  regression beyond 3%.

Two Wave32 groups now own one query tile at D=128. They split the QK head-
dimension chunks, reduce a bounded 2x16x16 partial score tile through 2 KiB of
additional LDS, share the online-softmax state, and split the PV output chunks
without a second reduction. The assembler result moves from 256 to 121 VGPRs,
removes 82 VGPR spills and 332 scratch bytes, and raises modeled occupancy from
6 to 12 waves/SIMD. Nine interleaved trials measure 2.045x at noncausal
`(1,16,1024,128)` and 2.106x at causal sequence 1009, with 100% win rates and
maximum differences of 5.6e-7 and 8.4e-6. D=64 and advanced GQA/window/bias/
soft-cap variants retain the one-wave kernel pending their own matrix.

### G6-C: split/reduced dK/dV backward attention

**Status: implemented and non-production; LLVM/MLIR 23 + ROCm 7.14
correctness reaffirmed, performance rejection revalidation open.**

- Separate dQ and dK/dV wave ownership.
- Reduce bounded partial dK/dV tiles in a second generated kernel.
- Measure `(1,8,512,64)` and `(1,16,1024,128)`, causal/noncausal and GQA.
- Promote only with at least 15% at D=128 and 10% at D=64; temporary storage
  must remain below one extra K+V gradient footprint.

The opt-in compiler candidate partitions query tiles across two dK/dV blocks,
writes the second split into exactly one extra dK+dV footprint, and launches a
generated reduction kernel. It matches the serial kernel across MHA, causal,
ragged, and GQA with maximum absolute error 3.13e-7. It does not clear the
performance gate: nine-trial device speedups span 0.908--1.013x and end-to-end
speedups span 0.904--1.070x, with neither required D=64 nor D=128 rung meeting
the gain threshold. The existing key-tile/head grid already exposes enough
parallelism, so the extra global partial traffic is not amortized. Production
therefore remains on serial-per-key-tile dK/dV; the candidate, correctness tests,
and rejection benchmark stay available for later architectures.

### Older-kernel retune closeout

The scalar f32 GEMM compiler now accepts compile-time output tiles. Production
uses 2x2 only for square sizes through 256 (0.0483 ms versus 0.0508 ms for the
old 4x4 at 256 cube) and conservatively retains 4x4 elsewhere where device and
end-to-end winners disagree. Grouped GEMM uses `tn=1` below 64k output elements,
`tn=2` at 64k, and `tn=4` from 131k, with exact-divisibility fallback; promoted
model rows are 1.92--3.42x faster in the resident kernel.

Grouped SwiGLU now collapses `3E` per-expert GEMM launches into three grouped
launches. E8 model rows improve resident GEMM time by 4.64--7.64x and end-to-end
time by 3.58--4.72x with 100% paired win rates. By contrast, compiler-generated
row-gather and weighted-scatter candidates for KV/MoE transport did not clear
both resident and end-to-end gates and remain non-production experiments.

The consolidated evidence is
`benchmarks/baselines/rocm_gfx1151_compiler_retune_2026_07_15.json`; the
reproducers are the `benchmark_rocm_{f32,grouped_gemm,swiglu,transport}_retune.py`
and `benchmark_rocm_g6{b_two_wave,c_split_reduced}.py` scripts.

All winners update
`benchmarks/baselines/rocm_gfx1151_hot_paths.json`; native counter collection
uses `benchmarks/rocm/collect_rocm6_counters.py` only on bare metal.

## Exact-device expansion queue

These retain the release priorities from `ROCM_AUDIT.md`, but execution is
hardware-gated. Compiler-only work may proceed locally; promotion waits for the
named device.

| ID | Priority | State | Target | Required first proof |
|---|---|---|---|---|
| ROCM-2 | P0 | open, access-gated | gfx1201, Radeon AI PRO R9700 | RDNA 4 matmul assembles, launches, and matches; establish WMMA-v2 fragment layout before adding FP8. |
| ROCM-1 | P0 | open, access-gated | gfx950, MI350 series | Compile, launch, and numerical proof for matmul, flash attention, softmax, and GELU; then CDNA 4 FP8/FP6/FP4 breadth. |
| ROCM-3 | P0 | open, access-gated | gfx1250, MI455X | Join the upstream-LLVM artifact to an exact-device matmul launch and numerical fixture. |
| ROCM-4b | P1 | open, access-gated | gfx942, MI300X/MI325X | Retain explicit compatibility proof for matmul, flash attention, softmax, and GELU. |
| ROCM-4a | P1 | open, access-gated | gfx1200, Radeon RX 9000 | Exact-device matmul proof plus stable rejection of unsupported feature forms. |
| ROCM-5 | P1 | landing; depends on rows above | all above | Close RDNA 4 WMMA-v2, gfx125x WMMA-v2, and CDNA MFMA descriptors with exact-device layouts, dtype guards, resources, and numerical proof. |

### P0 exact-device access coordination

The three P0 queues require externally scheduled hardware; no P0 device is
reachable from the gfx1151 WSL host. The access handoff is ready with one
common synchronization key, `ROCM-P0-LLVM23-2026-07`, and must retain the
configure cache, compiler versions, device identity, JUnit, emitted object,
and numerical outputs for each run.

| Queue | Required host | Access state | First scheduled command/result |
|---|---|---|---|
| ROCM-1 | MI350-series `gfx950` | owner and reservation required | LLVM/MLIR 23 clean build; matmul, flash attention, softmax, and GELU compile/launch/oracle packet |
| ROCM-2 | Radeon AI PRO R9700 `gfx1201` | owner and reservation required | LLVM/MLIR 23 clean build; WMMA-v2 fragment layout plus aligned/ragged matmul packet |
| ROCM-3 | MI455X `gfx1250` | owner and reservation required | LLVM/MLIR 23 clean build; upstream artifact joined to launch and numerical packet |

Access coordination is not complete until a named operator and reservation are
recorded for each host. Compiler-only artifacts may be prepared locally, but
no queue status advances from that evidence.

The LLVM 23 compiler-only handoff is prepared with
`benchmark_rocm_arch_fragments.py --artifact-directory`. The 2026-07-16 packet
contains input MLIR, target-lowered ROCDL, embedded code-object MLIR, resource
metadata, and median/MAD compiler timings for gfx1201, gfx950, gfx1250, gfx942,
and gfx1200. Every requested row assembled with a real target intrinsic, zero
scratch, and zero VGPR/SGPR spills. Recreate the transferable packet from the
clean build with:

```bash
TESSERA_OPT="$PWD/build-rocm-7.14-llvm23-clean/tools/tessera-opt/tessera-opt" \
MLIR_OPT=/usr/lib/llvm-23/bin/mlir-opt \
.venv/bin/python benchmarks/rocm/benchmark_rocm_arch_fragments.py \
  --repetitions 3 --arch gfx1201 --arch gfx950 --arch gfx1250 \
  --arch gfx942 --arch gfx1200 \
  --artifact-directory /tmp/tessera-rocm714-remote-packets \
  --output /tmp/tessera-rocm714-remote-packets.json
```

These packets are compiler-only. Remote operators must append device identity,
module load/launch, aligned and ragged numerical output, device and end-to-end
timing, and measured occupancy before any exact-target row advances.
The local bundle contains 40 files at
`/tmp/tessera-rocm714-remote-packets.tar.gz` with SHA-256
`3d569f1de9c837fefef5a84c435c5508f2f8d1c691c38620e59dd6a6a015ee4e`.

## ROCM-8: bare-metal copy versus zero-copy

**Status: blocked on access to a bare-metal gfx1151 host (2026-07-16). WSL
results are characterization only and cannot close ROCM-8.**

WSL measurements show an environment-specific crossover, but Windows driver
round trips affect registration and allocation. Before automatic selection:

1. collect copy and mapped-host measurements on bare-metal gfx1151;
2. report both kernel-only and end-to-end latency;
3. cover at least 256³ through 2,048³ GEMM plus representative serving buffers;
4. repeat enough samples to establish a stable crossover;
5. keep `TESSERA_ROCM_ZEROCOPY=1` opt-in unless the bare-metal evidence is
   reproducible and guarded by a ratchet.

## Accepted-deferred work

Consumer plan `SEQUENCE-MIXER-2026-07-17`: the compiler-direction Sequence Mixer
track ([`../../compiler/SEQUENCE_MIXER_ENGINEERING_PLAN.md`](../../compiler/SEQUENCE_MIXER_ENGINEERING_PLAN.md))
consumes ROCm as a **lead performance target** (Decision #28 — its WMMA/MFMA
candidates set the ceiling, never capped by the shared mixer framework). It
**extends already-complete vehicles** rather than opening new items: channel-wise
KDA/GDN decode → **ROCM-REPLAY-1** (add the channel-diagonal transition to the
proven persistent/flush/rollback/async-ring path); `windowed_kv` mixer state →
**ROCM-9** (window ring on the proven direct + gather paged-KV routes);
chunkwise-scan inner GEMMs → **ROCM-TILE-1** WMMA f16/bf16/int8/int4 fragments;
`sliding_window`/full mixer forward → **ROCM-6 G6-B**; mixer backward → **ROCM-6
G6-C** (split/reduced dK/dV). Low-precision mixer GEMMs stay bf16/f16 on gfx1151
per the standing **FP8/FP4 WMMA guard**; FP8/FP4 forms are the access-gated
CDNA4/RDNA4 packet (ROCM-1/2/3). Valid paired mixer device timing needs bare-metal
gfx1151 (ROCM-8). Inherits the exact-device native-provenance + aligned/ragged +
RDNA↛MFMA-guard contract unchanged. Direction pointer only; no ROCm gate or
exact-device claim changes here.

Cross-backend sync `EPILOGUE-CONTRACT-2026-07-16` updates only the shared
`FusedRegion` bias/activation/residual order and registered rejection
diagnostics. Existing gfx1151 epilogue fixtures already consume this oracle;
NVIDIA now validates the complete 43-case CUDA execution matrix. No CUDA warp,
register, dtype-support, or schedule result transfers to ROCm. Re-run the shared
contract and architecture-supported execution matrix on the gfx1151 host when
this coordinating change lands; retain explicit not-applicable results for
CUDA-only storage forms.

Cross-backend sync `NVIDIA-SM120-LOWP-2026-07-18`: not applicable to gfx1151
execution. The CUDA work changes no shared dtype/ScaleLayout, epilogue, or
autotune schema. gfx1151 supports neither FP8/BF8 WMMA nor NVIDIA NVFP4 OMMA;
ROCm therefore inherits no CUDA nibble packing, scale selector, wave schedule,
resource value, timing, or selector promotion. RDNA4/CDNA4 low-precision work
remains behind its architecture-specific exact-device queues.

Cross-backend sync `E2E-SPINE-2026-07-18`: ROCm owns **ROCM-E2E-1** and
**ROCM-E2E-2**. The shared contract standardizes native-image metadata and
launch descriptors only; ROCm continues to own directives, generators, AMDGPU
ISA selection, wave/LDS schedules, HSACO production, and selectors. The softmax
pilot must match the existing gfx1151 route before its runtime-authored text
bridge is removed. NVIDIA physical schedules do not transfer, and later
architecture breadth still requires the exact-device queues above. The
completed E2E-SPINE-0 foundation gives every registered ROCm family/exact target
a total family-shared pipeline row, including gfx1250, while preserving the
exact-target artifact fallback and generic runtime route. E2E-SPINE-1 adds the
portable HSACO image/descriptor envelope and registered validation diagnostics
without encoding AMD wave/LDS schedules or changing any HIP route. ROCM-E2E-1
remains responsible for the first typed softmax producer and exact gfx1151 join.
E2E-SPINE-2 completes the shared typed carriers, stage ledger, cache join, and
descriptor-first exact-target launcher registry. It registers no HIP hook and
does not replace runtime-authored directives or existing gfx1151 executors;
ROCM-E2E-1 still owns HSACO production, `gfx1151` registration/submission,
softmax comparison, cleanup, and the first ROCm Level-C row.
The NVIDIA-E2E-1 f16 landing slice was assessed as NVIDIA-only: it adds an
SM120 PTX package producer and exact CUDA submission hook, with no HIP hook,
ROCm directive/ABI, wave/LDS schedule, dtype registration, or selector change.
The completed NVIDIA-E2E-1 NVFP4 slice extends the shared `tile.matmul_kernel`
verifier with an explicit packed-A/packed-B/scale-A/scale-B/output/M/N/K form.
This is not applicable to the gfx1151 WMMA lowering because that ISA has no
NVFP4 block-scaled matrix instruction. ROCm inherits only shared verifier
rejection behavior; it does not inherit CUDA scale-word packing, warp geometry,
resources, timings, ABI registration, or selector evidence. RDNA4/CDNA4 work
remains in its architecture-owned exact-device queue.

The first NVIDIA-E2E-2 slice changes the shared Graph→Tile async contract so a
copy produces `!tile.async_token`, its wait retires that token, and a matrix
consumer carries the dependency. This is parity validated with ROCm's existing
token/retirement legality model and adds no ROCm directive, AMDGPU instruction,
wave/LDS schedule, HSACO ABI, selector, or execution claim. The additive
pipeline-registry driver-source field and `tessera_nvidia` dialect manifest row
are NVIDIA bookkeeping; ROCm pipeline ownership and runtime routing are
unchanged. Exact SM builders and CUDA TMA/WGMMA behavior are not applicable to
ROCm.

The NVIDIA-E2E-2 softmax slice adds the shared semantic
`tile.softmax_kernel(X, O, Rows, K)` envelope with explicit storage,
accumulation, and last-axis fields; the envelope now accepts f16/f32 storage
with f32 accumulation. ROCM-E2E-1 must assess producing or adapting
this envelope into the existing typed ROCm softmax directive/generator; that is
follow-up required, not parity inferred. ROCm does not inherit the SM120
thread-per-row schedule, `nvvm.ex2`, PTX ABI, resources, timings, or selector.
The existing cooperative HIP softmax execution path is unchanged.

The NVIDIA-E2E-2 dtype-totality slice changes the shared MMA selector contract:
fp32 Tensor Core selection now requires an explicit TF32 math mode, and bare
`fp4_e2m1` no longer aliases NVIDIA NVFP4. This semantic separation is parity
validated for ROCm: AMD xf32 continues to require its architecture-owned math
mode, and RDNA/CDNA FP4/MX scale formats remain distinct from UE4M3-scaled
NVFP4. The new SM120 scalar/vector and fragment table transfers no AMD ISA,
wave/VGPR layout, scale encoding, HSACO ABI, runtime readiness, or selector.

The follow-on SM120 dtype slice adds a backend-private
`tessera_nvidia.mx_block_scale_mma` Target IR op and ptxas-backed FP6/MXFP4
register contracts. This is not applicable to ROCm code generation and changes
no AMD dtype registry, MFMA/WMMA descriptor, scale encoding, HSACO ABI, runtime
route, or selector state. Exact CDNA/RDNA low-precision evidence remains owned
by the corresponding ROCm items.

The NVIDIA-E2E-2 reduction slice adds the shared semantic
`tile.reduce_kernel(X, O, Outer, AxisExtent, Inner)` envelope with explicit
kind, storage, accumulation, normalized axis/keepdims, schedule, and NaN
policy. ROCM-E2E-2 must assess adapting that
carrier to the existing HIP reduction generator; this is follow-up required,
not parity inferred. ROCm inherits neither the CUDA serial nor cooperative-128
schedule, PTX ABI, resource/timing evidence, runtime readiness, or selector
change.

The NVIDIA-E2E-2 epilogue slice tightens the shared `tile.matmul_kernel`
verifier around optional residual operands and the portable
matmul/bias/activation/residual order. ROCm semantic parity is preserved, but
its materializer must opt into that launch form explicitly; no CUDA buffer
layout, warp schedule, PTX ABI, resources, timing, or readiness transfers.

The NVIDIA-E2E-2 attention slice adds the shared semantic
`tile.attention_kernel(Q,K,V,O,B,Hq,Hkv,Sq,Sk,D,Dv)` carrier with explicit
storage, f32 accumulation/output, scale, and causal policy. ROCM-E2E-2 must
assess adapting it to the existing HIP/MFMA forward routes; this is follow-up
required, not parity inferred. ROCm inherits no CUDA thread-per-output
schedule, PTX ABI, resource/timing evidence, readiness, or selector change.

The NVIDIA paged-KV slice adds a shared logical-page gather carrier with f32
pages, i32 table, explicit dimensions/range, and a direct-route semantic tag.
ROCM-E2E-2 must map it to the existing HIP paged movement lane under its own
ISA and exact-device proof; no PTX ABI, CUDA schedule, timing, or selector
state transfers.

The NVIDIA backward-attention slice adds a shared
`tile.attention_backward_kernel` semantic carrier with explicit mask, softcap,
determinism, route, and workspace fields. ROCM-E2E-2 must assess mapping it to
the existing compiler-generated HIP/WMMA backward sequence. The CUDA
single-owner schedule, zero-workspace reference ABI, atomic/split resources,
timings, and selector evidence do not transfer to gfx1151.

Cross-backend sync `E2E-DEVICE-LIBS-2026-07-19` extends the shared native-image
contract with content-addressed LLVM-stage device-library provenance. ROCm must
populate it from the matching clang/ROCm-driver-selected OCML, OCKL, and OCLC
set under `--rocm-path`; it must not copy NVIDIA's explicit single-libdevice
link rule or hand-assemble an OCLC set independently of architecture, wavefront,
and math-mode flags. ROCM-E2E-1 now closes that follow-up for the gfx1151
softmax pilot: AMD clang selects the seven records, their content digests enter
cache/toolchain identity, and absolute paths stay out of the artifact. The
existing LLVM 23/TheRock 7.14 compatibility rule remains mandatory, and no gfx
execution or selector state changes in the NVIDIA-owned landing.

Cross-backend sync `CUDA-MATH-CONTRACT-2026-07-19` makes the shared Tile softmax
envelope state its exponential mode and FTZ behavior instead of deriving either
from compiler optimization flags. ROCm must map that semantic choice to an
architecture-owned OCML/intrinsic route and validate its own accuracy and
denormal behavior under ROCM-E2E-1; PTX's `ex2.approx.f32` bound is not AMD
evidence and no gfx selector changes here.

Cross-backend sync `CUDA-INTRINSIC-SURFACE-2026-07-19` extends the shared
rounding vocabulary with toward-positive and toward-negative while preserving
the existing default RTNE/RTNA/RTZ tuning sweep. This is shared semantic parity;
ROCm must map directed conversions to AMDGPU/OCML behavior under its own typed
lowering and exact-device proof. CUDA integer, DP2A/DP4A, cast-suffix, and packed
SIMD symbols transfer no AMD instruction, wave layout, runtime route, or
selector. The NVIDIA inventory marks all Tessera Target-IR/runtime rows planned.

Cross-backend sync `PTX-TYPE-MEMORY-TRUTH-2026-07-19` adds physical PTX register
and format-kind fields to the NVIDIA dtype contract and a backend-private PTX
memory-model guard. This is not AMD ISA evidence. ROCm must continue deriving
VGPR packing, alternate formats, scopes, atomics, and cache/coherence behavior
from the applicable RDNA/CDNA ISA and ROCm device-library contract. The only
shared outcome is the architectural rule that language dtype availability does
not imply a native register type or executable matrix route.

Cross-backend sync `NVIDIA-E2E-DTYPE-EXEC-2026-07-19` extends the shared Tile
epilogue output vocabulary with f64 for NVIDIA's architecture-owned m8n8k4
DMMA route. ROCm records parity at the shared semantic layer only: AMD
MFMA/WMMA lane maps, packed formats, code-object ABI, timings, and selectors do
not inherit CUDA evidence. Existing ROCm f64 states still require exact-gfx
proof.

Cross-backend sync `X86-E2E1-NATIVE-CPU-2026-07-19` classifies shared native
descriptor results for `cpu`, `x86`, `x86_amx`, and `x86_avx512` as
`native_cpu` with CPU-wall timing rather than GPU-event timing. ROCm artifact,
ABI, stream, event, HSACO, device-library, exact-gfx, and selector contracts are
unchanged. The x86 pilot consumes existing Tile softmax/reduction carriers and
adds no shared dtype, operation, or verifier state that ROCm must implement.

Cross-backend sync `X86-E2E1-BREADTH-2026-07-19` consumes the already-shared
matmul and attention launch carriers for x86 f32 GEMM and MHA descriptors.
ROCm inherits no AVX-512 ABI, host schedule, timing, readiness, or selector
state; its WMMA/MFMA and attention routes remain architecture-owned. The x86
restriction to equal query/KV head counts and zero dropout is not a shared
semantic restriction and changes no ROCm verifier or capability row.

Do not schedule these without new evidence:

- flash-attention K/V double buffering, until a measured workload or target is
  staging-bound rather than occupancy/LDS-bound;
- packed-memory int4 GEMM on gfx1151 as a compute optimization—its value there
  is footprint/bandwidth, not higher matrix issue rate;
- automatic zero-copy selection before ROCM-8 bare-metal measurements;
- FP8/BF8 WMMA on gfx1151, which is an unsupported instruction claim rather
  than an optimization opportunity.

## Validation and update checklist

Run tests on the host ROCm environment with the intended GPU and toolchain
visible. A sandbox fallback cannot count as device proof.

- Tile/WMMA/MFMA:
  `test_rocm_target_wmma_lowering.py`, `test_rocm_wmma_gemm_via_mlir.py`, and the
  new portable fragment fixtures.
- Paged serving: `test_paged_kv_rocm_abi.py` and
  `test_paged_kv_rocm_native.py`.
- GEMM/attention ratchets: `test_rocm_perf_ratchet.py` and the corresponding
  benchmark recorder.
- Exact-target dashboards:
  `python -m tessera.cli.gpu_target_map --target=rocm --render` and
  `python -m tessera.cli.conformance_matrix --render`.
- Static gates: mypy ratchet, Ruff, and `git diff --check`.
- After changes: `graphify update .`.

## Definition of ROCm roadmap closure

The ROCm roadmap is not closed merely because gfx1151 has broad operator
coverage. Closure requires:

- portable Tile fragments executing on at least one RDNA WMMA and one CDNA MFMA
  target;
- stable paged-KV serving with exact-device proof and measured route selection;
- ReplaySSM persistent/asynchronous serving parity;
- ROCM-6 candidates revalidated under the current toolchain and either promoted
  through their ratchets or explicitly retained as measured non-winners;
- exact-device evidence for the priority RDNA 4/CDNA targets, without inherited
  gfx1151 proof;
- generated target, runtime, and conformance dashboards agreeing with the
  checked-in evidence.

Cross-backend sync `E2E-SPINE-2026-07-18` records the 2026-07-20 scoped x86
selector retirement: eligible static X86-E2E-1 modules now use their canonical
descriptor by default. ROCm parity is not applicable; no ROCm pipeline, HSACO
ABI, schedule, dtype capability, or selector changes. X86-E2E-2 owns the
remaining AVX-512 inventory and must reassess ROCm only when a shared contract
changes.

Cross-backend sync `X86-E2E2-ELEMENTWISE-2026-07-20` adds the internal shared
`tile.elementwise_kernel` semantic carrier for f32 unary/binary and f32-to-bool
predicate requests. ROCm parity is assessed at the carrier boundary only;
AVX-512 ABIs, host schedules, CPU-wall evidence, and the 16K binary selector
threshold transfer no RDNA instruction, HSACO ABI, exact-gfx evidence, runtime
readiness, or selector claim. Existing ROCm elementwise routes remain
architecture-owned and unchanged.

Cross-backend sync `X86-E2E2-TYPED-LOGIC-2026-07-20` widens that internal
carrier with compare, logical, and bitwise semantics plus explicit f32/i8/i32
physical storage. The capability repair is x86-owned bool/int32 truth for
already-shipped AVX-512 ABIs. ROCm inherits no x86 C ABI, null-operand
convention, 32K selector threshold, CPU timing, RDNA instruction, HSACO route,
or selector claim; ROCm target and execution rows remain unchanged.

Cross-backend sync `X86-E2E2-FLAT-FOLLOWON-2026-07-20` extends the shared
elementwise carrier with where, transcendental, and binary-math semantics.
ROCm parity is assessed at the carrier boundary only; AVX-512 approximations,
C ABIs, CPU-wall thresholds, exact-host evidence, RDNA instructions, HSACO
routes, and ROCm selectors do not transfer.

Cross-backend sync `X86-E2E2-DTYPE-2026-07-20` adds an x86-only datatype/CPUID
contract and BF16, VNNI U8/S8, and FP64 descriptor ABIs. ROCM-DTYPE-1 remains
the independent AMD GPU authority; no gfx capability, packing, accumulator,
execution, or selector row changes.

Cross-backend sync `ATTN-DIALECT-MLIR23-2026-07-20` corrects the internal MLIR
attention dialect namespace from the nested `tessera.attn` spelling to the
MLIR-23-compatible `tessera_attn` spelling. Public Graph IR operation names,
attention semantics, ROCm target capabilities, HSACO ABIs, schedules, and
selector state are unchanged; ROCm parity is validated by the shared attention
lit coverage.

Cross-backend sync `LLVM23-BACKBONE-2026-07-20` makes LLVM/MLIR 23.x the sole
accepted compiler build environment. Top-level and standalone CMake entry
points reject every other major and mixed installations; ROCm uses the
versioned apt LLVM 23 packages with ROCm 7.14. ROCm target semantics, HSACO
ABIs, and selectors are unchanged; host-free compiler/lit and gfx1151 unit
proofs validate parity without transferring evidence to another AMD target.
