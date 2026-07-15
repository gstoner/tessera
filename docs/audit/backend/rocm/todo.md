---
last_updated: 2026-07-14
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

## Recommended implementation order on gfx1151

| Order | ID | Work | Why now | Completion gate |
|---:|---|---|---|---|
| 1 | ROCM-TILE-1 | Portable Tile fragment materialization for RDNA WMMA | It closes the remaining cross-vendor Tile IR gap and provides the reusable boundary for later architectures and dtypes. | The same logical Tile fixture resolves, packs, executes, unpacks, stores, and matches the reference on gfx1151 without test-authored physical fragments. |
| 2 | ROCM-9 | Exact-device paged-KV proof and measured fused consumer | The stable ABI exists, but its non-identity HIP fixture still needs exact-device closure; direct page-table attention should land only if it beats gather→FA. | Permuted pages execute on gfx1151; gather→FA and direct-paged candidates match the same oracle; device and end-to-end rows select a winner without changing the ABI. |
| 3 | ROCM-REPLAY-1 | Persistent ROCm ReplaySSM serving path | Selective SSM exists, but ROCm lacks CUDA-equivalent persistent S0, output-only decode, flush, and asynchronous serving machinery. | Long decode, flush, rollback, speculative rejection, block submit, and an ordered async ring match `SSMStateHandle`; traffic and latency are committed. |
| 4 | ROCM-6 | Run the three ratcheted gfx1151 redesign experiments | Existing measurements identify occupancy/VGPR ownership—not another generic staging layer—as the credible performance lever. | Each candidate clears its named A/B rungs and regression thresholds before replacing production. |
| 5 | ROCM-5 | Generalize fragment/layout selection by architecture | RDNA 4 WMMA v2 and CDNA MFMA cannot reuse gfx1151 register maps. | Per-family descriptor, pack/unpack, legality guards, and exact-device fixtures exist. |

## ROCM-TILE-1: portable Tile fragments

The portable dialect already defines `!tile.fragment`, `#tile.mma_desc`,
`tile.view`, `tile.fragment_pack`, `tile.mma`, `tile.fragment_unpack`, and
`tile.store`. NVIDIA consumes the typed form; ROCm currently rejects it before
physical materialization.

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

## ROCM-9: paged-KV serving

The ABI is fixed and portable:

- K/V pages: physical `[P, L, H, D]`;
- page table: i32 logical-page to physical-page mapping;
- token indices: i64 gather order;
- attention must not assume identity or contiguous page placement.

### Work

1. Run `test_live_rocm_paged_gather_handles_permuted_pages` on gfx1151 and
   retain the actual architecture provenance.
2. Extend the numerical fixture to cover a non-identity table, arbitrary token
   order, multi-query causal offsets, and page crossings in one case.
3. Retain gather→dense-FA as the baseline.
4. Build a direct paged-attention candidate that consumes the same page table
   inside K/V traversal without materializing dense K/V.
5. Event-time both candidates over decode shapes such as 128, 512, 2,048, and
   8,192 cached tokens; also report end-to-end latency.
6. Persist the crossover in the ROCm D2 corpus and warm-start runtime selection.

The fused candidate is not automatically the production winner. Short contexts
may benefit from removing gather buffers, while long contexts may need the
parallelism of the staged FA path.

Evidence starts in:

- `tests/unit/test_paged_kv_rocm_abi.py`;
- `tests/unit/test_paged_kv_rocm_native.py`;
- `python/tessera/compiler/emit/rocm_hip.py`;
- `python/tessera/cache/paged_kv.py`.

## ROCM-REPLAY-1: ReplaySSM serving parity

The reference state ABI, flush policy, speculative rollback, and CUDA serving
implementation already define the semantics. The ROCm work must preserve those
semantics rather than introduce a backend-specific cache layout.

### Work

1. Add a HIP-owned persistent S0 and fixed-capacity replay-input ring.
2. Implement scalar-A output-only decode without writing the full `[B,D,N]`
   state each token.
3. Materialize S0 only at the existing flush boundary.
4. Add block submission for prefill/speculative verification.
5. Add a multi-slot ordered asynchronous ring with pinned staging, HIP streams,
   timing events, and opaque device-output leases.
6. Support device-resident consumers before D2H; host output remains an explicit
   wait/download operation.
7. Benchmark summary traffic against ReplaySSM analytical traffic and record
   device latency, wall latency, throughput, flush frequency, and ring pressure.

### Required proof

- long decode across multiple flushes versus `SSMStateHandle`;
- rollback and speculative rejection equivalence;
- output-only and state-and-output route equivalence;
- ring ordering, backpressure, event waits, and device-buffer lifetime tests;
- representative serving rows committed to D2 with exact gfx1151 provenance.

## ROCM-6: performance redesign experiments

Production remains in place while each candidate is measured.

### G6-A: VGPR-bounded multi-wave GEMM

- Split an output macro-tile across two Wave32 groups.
- Reduce bounded partial f32 accumulators through LDS.
- Keep per-wave accumulator pressure below the measured 4x4 VGPR cliff.
- Measure f16/bf16 at 2048³ and 4096³, ragged
  `2049x4093x2051`, and int8 at 2048³.
- Promote only with at least 10% median gain on both aligned f16 rungs, no rung
  slower by more than 3%, and all dtype/ragged oracles green.

### G6-B: two-wave online-softmax forward attention

- Give two waves one query tile and share K/V traversal.
- Merge per-wave online `(m,l,O)` state once per K/V tile.
- Measure `(1,8,512,64)`, `(1,8,1024,64)`, `(1,16,1024,128)`, and causal
  sequence 1009 at D=128.
- Promote only with at least 10% gain on both D=128 rungs and no D=64
  regression beyond 3%.

### G6-C: split/reduced dK/dV backward attention

- Separate dQ and dK/dV wave ownership.
- Reduce bounded partial dK/dV tiles in a second generated kernel.
- Measure `(1,8,512,64)` and `(1,16,1024,128)`, causal/noncausal and GQA.
- Promote only with at least 15% at D=128 and 10% at D=64; temporary storage
  must remain below one extra K+V gradient footprint.

All winners update
`benchmarks/baselines/rocm_gfx1151_hot_paths.json`; native counter collection
uses `benchmarks/rocm/collect_rocm6_counters.py` only on bare metal.

## Exact-device expansion queue

These retain the release priorities from `ROCM_AUDIT.md`, but execution is
hardware-gated. Compiler-only work may proceed locally; promotion waits for the
named device.

| ID | Priority | Target | Required first proof |
|---|---|---|---|
| ROCM-1 | P0 | gfx950, MI350 series | Compile, launch, and numerical proof for matmul, flash attention, softmax, and GELU; then CDNA 4 FP8/FP6/FP4 breadth. |
| ROCM-2 | P0 | gfx1201, Radeon AI PRO R9700 | RDNA 4 matmul assembles, launches, and matches; establish WMMA v2 fragment layout before adding FP8. |
| ROCM-3 | P0 | gfx1250, MI455X | Join the upstream-LLVM artifact to an exact-device matmul launch and numerical fixture. |
| ROCM-4a | P1 | gfx1200, Radeon RX 9000 | Exact-device matmul proof plus stable rejection of unsupported feature forms. |
| ROCM-4b | P1 | gfx942, MI300X/MI325X | Retain explicit compatibility proof for matmul, flash attention, softmax, and GELU. |
| ROCM-5 | P1 | all above | Separate RDNA 4 WMMA v2 and CDNA MFMA descriptors, fragment layouts, and dtype guards. |

## ROCM-8: bare-metal copy versus zero-copy

WSL measurements show an environment-specific crossover, but Windows driver
round trips affect registration and allocation. Before automatic selection:

1. collect copy and mapped-host measurements on bare-metal gfx1151;
2. report both kernel-only and end-to-end latency;
3. cover at least 256³ through 2,048³ GEMM plus representative serving buffers;
4. repeat enough samples to establish a stable crossover;
5. keep `TESSERA_ROCM_ZEROCOPY=1` opt-in unless the bare-metal evidence is
   reproducible and guarded by a ratchet.

## Accepted-deferred work

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
- ROCM-6 candidates either promoted through their ratchets or explicitly
  retained as measured non-winners;
- exact-device evidence for the priority RDNA 4/CDNA targets, without inherited
  gfx1151 proof;
- generated target, runtime, and conformance dashboards agreeing with the
  checked-in evidence.
