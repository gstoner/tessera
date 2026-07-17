---
last_updated: 2026-07-16
audit_role: plan
plan_state: open
scope: ROCm backend implementation and exact-device proof
---

# ROCm backend TODO

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

## LLVM/MLIR 23 and ROCm 7.14 transition evidence

**Status: host build and gfx1151 correctness ratchets complete in WSL
(2026-07-16); bare-metal-only gates remain open.**

The project-wide compiler floor is now a matched LLVM/MLIR 23 toolchain. On the
gfx1151 WSL host, the validated configuration uses upstream Ubuntu LLVM/MLIR
23.0.0 for Tessera's C++ compiler and TheRock Core SDK 7.14 for HIP, HIPRTC,
device libraries, and the HIP compiler. Mixing the former LLVM/MLIR 22 build
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
| LLVM23/ROCm 7.14 | complete on gfx1151 WSL | Clean build, 32/32 ROCm lit, 1280/1280 compiled correctness, and 21/21 valid performance ratchets pass; the combined sweep is 86/90 with four zero-event-only failures. |
| ROCM-TILE-1 | complete on gfx1151 | Portable f16/bf16/int8/int4 fragments execute and compare on gfx1151. Other architectures are owned by ROCM-1 through ROCM-5. |
| ROCM-9 | complete on gfx1151 | Non-identity paged-KV direct and gather routes execute, compare, and have a measured serving decision. |
| ROCM-REPLAY-1 | complete on gfx1151 | Persistent state, flush/rollback, block submission, asynchronous ring, lifetime proof, and the wider performance matrix are committed. |
| ROCM-6 | open revalidation; timing blocked | LLVM/MLIR 23 + ROCm 7.14 correctness is green for G6-A/B/C, but WSL HIP events return invalid zero durations. Existing production choices stay in force pending valid paired device timing. |
| ROCM-8 | blocked | Bare-metal gfx1151 access is required; WSL characterization cannot close it. |
| ROCM-1/2/3 | open, access-gated | P0 exact-device execution on gfx950, gfx1201, and gfx1250 is the active release frontier. |
| ROCM-4a/4b | open, access-gated | P1 compatibility execution on gfx1200 and gfx942 follows the P0 packet. |
| ROCM-5 | landing, exact-device closure open | Architecture-owned descriptors and cross-assembly exist; numerical and performance closure depends on ROCM-1 through ROCM-4b. |

## Recommended open-work order

Completed and measured-non-winning gfx1151 work is intentionally absent from
this queue. The local WSL host may prepare artifacts and harnesses, but only the
named exact device can satisfy an execution gate.

| Order | ID | Work | Access state | Completion gate |
|---:|---|---|---|---|
| 1 | ROCM-2 | Run the common P0 packet on Radeon AI PRO R9700 `gfx1201` | owner and reservation required | RDNA 4 WMMA-v2 f16/bf16 plus enabled FP8/integer forms assemble, launch, match aligned/ragged oracles, and record resources and timing. |
| 2 | ROCM-1 | Run the common P0 packet on MI350-series `gfx950` | owner and reservation required | CDNA 4 matmul, flash attention, softmax, and GELU launch and compare; low-precision breadth advances only with physical-layout proof. |
| 3 | ROCM-3 | Run the common P0 packet on MI455X `gfx1250` | owner and reservation required | The upstream-LLVM artifact joins to a launch/numerical proof; WMMA-v2 properties and fragment layout match the device. |
| 4 | ROCM-6 | Revalidate G6-A/B/C with valid paired device timing | bare-metal gfx1151 or repaired event timing required | Original correctness, resource, aligned/ragged, dtype, device-time, and E2E gates are rerun under LLVM/MLIR 23 + ROCm 7.14 before reaffirming or changing production. |
| 5 | ROCM-8 | Measure copy versus mapped-host memory on bare-metal `gfx1151` | bare-metal owner and reservation required | Repeated kernel-only and end-to-end measurements establish a stable crossover without using WSL evidence. |
| 6 | ROCM-4b | Retain compatibility proof on MI300X/MI325X `gfx942` | owner and reservation required | f16/bf16 MFMA plus retained matmul/attention/softmax/GELU paths launch and compare. |
| 7 | ROCM-4a | Add Radeon RX 9000 `gfx1200` exact-device proof | owner and reservation required | Matmul launches and compares; unsupported forms reject stably. |
| 8 | ROCM-5 | Close the architecture-owned fragment umbrella | depends on ROCM-1 through ROCM-4b | Every enabled family/dtype has exact-device packing, numerical, resource, and timing evidence, or an explicit unsupported/deferred state. |
| 9 | ROCM-TEST-1 | Validate ROCm host-free compiler ownership | available on the ROCm build host | Build the declared ROCm compiler capability set, explicitly select or exclude foreign-backend compiler tests, and retain command, build flags, tool path, node IDs, and diagnostics; no CUDA/Apple-only build assumption blocks the ROCm lane. |

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
flash-attention ladders with ROCm 7.2 and LLVM 22. This changes the premise of
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

Cross-backend sync `EPILOGUE-CONTRACT-2026-07-16` updates only the shared
`FusedRegion` bias/activation/residual order and registered rejection
diagnostics. Existing gfx1151 epilogue fixtures already consume this oracle;
NVIDIA now validates the complete 43-case CUDA execution matrix. No CUDA warp,
register, dtype-support, or schedule result transfers to ROCm. Re-run the shared
contract and architecture-supported execution matrix on the gfx1151 host when
this coordinating change lands; retain explicit not-applicable results for
CUDA-only storage forms.

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
