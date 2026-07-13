---
last_updated: 2026-07-13
audit_role: sub_audit
---

# ROCm Backend Audit

This audit answers three questions:

1. What ROCm functionality executes today, and on which exact device?
2. What evidence earns each claim?
3. What work remains, in priority order?

It is intentionally not a release-count dashboard or a chronological change
log. Detailed Strix Halo bring-up history lives in
[`STRIX_HALO_EXECUTION_PLAN.md`](STRIX_HALO_EXECUTION_PLAN.md), and reusable AMD
design guidance lives in
[`ROCM_PATTERNS_FROM_AMD_ECOSYSTEM.md`](ROCM_PATTERNS_FROM_AMD_ECOSYSTEM.md).

> **Status authority.** ROCm proof is recorded at exact-target granularity. The
> generic `rocm` name is a family rollup and never inherits compile, execution,
> numerical, or performance proof. Use the generated
> [`rocm_target_map.md`](../../generated/rocm_target_map.md) for per-feature
> status and [`runtime_execution_matrix.md`](../../generated/runtime_execution_matrix.md)
> for executable compiler/runtime lanes. Generated dashboards own counts; this
> document explains capabilities, evidence, decisions, and remaining work.

## Current status

The live proof target is `rocm_gfx1151`: RDNA 3.5 / Wave32 WMMA on a Ryzen AI
Max+ 395. On that target, the curated matrix, normalization, positional, and
attention feature groups have device execution and numerical evidence. Proof
does not transfer to RDNA 4, Wave32 WMMA v2, or CDNA MFMA targets.

| Feature surface | Real today on `rocm_gfx1151` | Remaining frontier | Authority |
|---|---|---|---|
| Matrix compute | `matmul` has shipped C-ABI proof; `batched_gemm`, `einsum`, `factorized_matmul`, `linear_general`, `qkv_projection`, and fused epilogues execute through compiler-generated WMMA lanes. | Exact-device proof on other architectures; occupancy and issue-rate tuning. | [`rocm_target_map.md`](../../generated/rocm_target_map.md) |
| Dense attention | `flash_attn` has shipped C-ABI proof. Compiler-generated forward/backward, GQA, MQA, MHA, sliding-window attention, additive bias, and logit soft-capping execute through the FA lane. | Higher-occupancy kernel design and broader exact-target validation. | [`rocm_target_map.md`](../../generated/rocm_target_map.md), [`runtime_execution_matrix.md`](../../generated/runtime_execution_matrix.md) |
| Linear, exotic, sparse, and recurrent attention | Linear/lightning attention, `gated_attention`, MLA decode variants, `hybrid_attention`, `deepseek_sparse_attention`, and the DeltaNet family have executing compiler-generated lanes. | Cooperative sparse tiling and larger device-resident selection. | [`rocm_target_map.md`](../../generated/rocm_target_map.md) |
| Norms and activations | `softmax`/`softmax_safe`, `rmsnorm`/`rmsnorm_safe`, `layer_norm`, `gelu`, `silu`, and `silu_mul` execute as compiler-generated kernels. | Optimize only where measurements identify a useful rung. | [`rocm_target_map.md`](../../generated/rocm_target_map.md) |
| Positional features | `rope` and `alibi` execute as compiler-generated kernels. | Exact-device proof on additional architectures. | [`rocm_target_map.md`](../../generated/rocm_target_map.md) |
| General math and reductions | Row reductions, unary/binary math, comparisons, logical operations, and bitwise operations have native gfx1151 lanes. | Broaden tuned shapes and dtype coverage without weakening numerical semantics. | [`runtime_execution_matrix.md`](../../generated/runtime_execution_matrix.md) |
| SSM, MSA, and EBM | Selective SSM forward/backward, MSA sparse attention, and native EBM kernels execute on gfx1151. | Optimize broader shapes and promote only with exact-target provenance. | [`runtime_execution_matrix.md`](../../generated/runtime_execution_matrix.md) |
| Asynchronous staging | Global→LDS `async_copy` and a staged WMMA GEMM tile execute correctly. | Use the substrate in production only when profiling finds a staging-bound kernel. | `test_rocm_async_copy_runnable.py`, `test_rocm_gemm_staged_async_copy.py` |

## Open actions

The table below is the actionable ROCm backlog. An item is complete only when
its stated evidence exists; emitting an artifact alone is not completion.

| ID | Priority | Action | Completion evidence |
|---|---|---|---|
| ROCM-1 | P0 | Add gfx950 MI350-series exact-device proof for the currently advertised artifact rows. | Matching gfx950 compiler, launch, numerical fixture, and `evidence_arch`; generated rows promote without inheriting gfx1151 data. |
| ROCM-2 | P0 | Add gfx1201 Radeon AI PRO R9700 exact-device proof. | At minimum, the current matmul artifact assembles, launches, and matches its numerical oracle on gfx1201. |
| ROCM-3 | P0 | Add gfx1250 MI455X exact-device proof. | The upstream-LLVM artifact is joined to an exact-device launch and numerical fixture with gfx1250 provenance. |
| ROCM-4 | P1 | Add gfx1200 consumer-device proof and retain gfx942 as an explicitly tested compatibility target. | Each promoted row carries its own runtime and numerical evidence; unsupported feature forms fail with stable diagnostics. |
| ROCM-5 | P1 | Finish architecture-specific MMA enablement instead of reusing gfx1151 layouts. | Separate RDNA 4, Wave32 WMMA v2, and CDNA MFMA fragment/layout guards pass assemble-and-compare fixtures on their matching devices. |
| ROCM-6 | P1 | Raise GEMM and flash-attention roofline attainment on gfx1151. | Device-timed ratchets show a repeatable gain without correctness, ragged-shape, or dtype regressions. Focus on VGPR/LDS budgets, occupancy, and WMMA issue scheduling. |
| ROCM-7 | P1 | Promote sparse-attention performance. | Cooperative tiled sparse attention and a larger GPU-resident top-k path beat the current baseline on representative DK2/MSA workloads and retain dense-equivalence checks. |
| ROCM-8 | P2 | Re-evaluate copy versus zero-copy on bare-metal ROCm. | Device and end-to-end measurements identify a stable crossover outside WSL before any automatic selection policy lands. |

### Accepted-deferred work

These are measured decisions, not unowned tasks:

- **Flash-attention KV double buffering:** defer until a kernel or device is
  staging-bound. Current gfx1151 kernels are occupancy/LDS-bound, and the
  runnable `async_copy` substrate is already available.
- **Packed-memory int4 GEMM:** defer on gfx1151. Int4 has no matrix-rate advantage
  over f16/bf16 there; packing would target memory footprint and bandwidth, not
  compute throughput.
- **Automatic zero-copy selection:** defer until bare-metal data replaces the
  WSL-specific crossover measurements.

## Proof environment and target semantics

The primary development system is Ubuntu 24.04 under WSL2 with ROCm 7.2.4 and
LLVM/MLIR 22.1.8. The device enumerates natively as `gfx1151`. Early bring-up
temporarily reported `gfx1100`; historical notes that name gfx1100 describe that
environment and are not evidence for a separate exact target.

The gfx1151 WMMA surface intentionally includes f16, bf16, iu8, and iu4 forms.
It does not include the FP8/BF8 and expanded WMMA forms found in newer families.
CDNA targets use MFMA and require different feature, dtype, and fragment-layout
proof. The exact-target map therefore leaves unexecuted rows `artifact_only`.

Current target priorities are:

1. gfx950, marked current datacenter;
2. gfx1250, marked forward datacenter;
3. gfx1201 and gfx1200, the current workstation/consumer RDNA 4 targets;
4. gfx1151, the proven development target;
5. gfx942, retained for compatibility.

Priority describes where proof should be added; it does not imply that one
architecture may borrow another architecture's evidence.

## Execution and compiler-path closure

The ROCm path progressed from artifact emission to a compiler-generated kernel
that launches through the production runtime. The completed milestones are
summarized here; the full chronology remains in the Strix Halo execution plan.

| Milestone | Result | Representative evidence |
|---|---|---|
| Target selection | `gfx11xx` lowers to WMMA; CDNA lowers to MFMA; unsupported dtype/arch combinations are gated. | ROCm target and lowering fixtures |
| Assemble | LLVM/ROCDL lowers to real AMDGPU objects containing `v_wmma_*`. | `test_rocdl_emit.py` |
| C-ABI launch | A WMMA GEMM launches through the registered HIP bridge and matches a host oracle. | `test_runtime_abi_rocm_launch_bridge.py`, `test_rocm_wmma_execute_compare.py` |
| Shipped ABI kernels | GEMM and flash-attention runtime symbols execute and compare on gfx1151. | `test_rocm_wmma_runtime_symbol.py`, `test_rocm_flash_attn_runtime_symbol.py` |
| MLIR→hsaco→execute | A Tessera kernel lowers through ROCDL, serializes to hsaco, loads through HIP, and executes. | `test_rocm_mlir_to_hsaco.py` |
| Real Target-IR WMMA | `tessera_rocm.wmma` lowers to the matching ROCDL/LLVM intrinsic with real RDNA fragments. | `test_rocm_target_wmma_lowering.py`, `test_rocm_wmma_gemm_via_mlir.py` |
| Generated GEMM | The compiler expands `tessera_rocm.wmma_gemm` into a fragment-materialized kernel rather than relying on authored HIP source. | `test_rocm_wmma_gemm_generated.py` |
| General production lane | Generated GEMM supports tiled K loops, ragged shapes, f16/bf16/int8/int4, in-process serialization, caching, and `runtime.launch()`. | `test_rocm_wmma_gemm_general.py`, `test_rocm_compiled_launch_execute.py` |
| Frontend ownership | Graph/Tile lowering emits the executable WMMA GEMM directive consumed by the generator. | `test_rocm_matmul_front_end_glue.py`, `test_target_ir_contract.py` |
| Runnable async staging | `tessera_rocm.async_copy` lowers to executable global→LDS movement and composes with WMMA compute. | `test_rocm_async_copy_runnable.py`, `test_rocm_gemm_staged_async_copy.py` |

The hand-written HIPRTC GEMM remains a shipped C-ABI oracle and availability
fallback. The compiler-generated lane is the default when its toolchain and
device prerequisites are present. A genuine compiled-kernel failure is surfaced;
fallback is reserved for an unavailable compiler/runtime path, not for masking a
miscompile.

## Landed feature evidence

This section records implementation shape only where it clarifies what the
feature claim means. Individual status rows and counts remain generated.

### Matrix and attention

- **General WMMA GEMM:** one generated kernel handles aligned, rectangular, and
  ragged M/N/K shapes. Its interior path uses contiguous vector loads; edge and
  K-tail paths retain masking without placing per-element masking in the common
  aligned K loop.
- **Fused epilogues:** optional column bias and relu/gelu/silu execute on the f32
  accumulator before the store. Integer inputs reject unsupported epilogues with
  a structured error.
- **Flash attention:** compiler-generated forward and backward use the same RDNA
  WMMA primitive, f32 softmax/accumulation, causal/ragged handling, and
  `runtime.launch()`. GQA/MQA share the forward body; grouped backward combines
  shared-KV gradients correctly.
- **Attention variants:** sliding windows skip fully excluded KV tiles; logit
  soft-capping applies `cap*tanh(score/cap)` before masking and softmax; additive
  bias provides the DFlash seam.
- **Linear attention:** identity, relu, and degree-2 feature maps execute without
  softmax. Lightning/retention variants add causal decay.
- **Composite attention:** gated attention and MLA variants compose already
  proven GEMM, attention, and pointwise lanes. DeltaNet uses a dedicated causal
  sequential-scan kernel rather than claiming composition as a new fused kernel.

### Norms, reductions, and general math

- **Softmax and reductions:** row kernels use f32 reduction regardless of
  storage dtype. Softmax uses stable max subtraction; layer norm uses a stable
  two-pass squared-deviation calculation rather than `E[x²]-E[x]²`.
- **Elementwise math:** unary/binary transcendental paths lower through
  math→ROCDL/OCML. Maximum/minimum preserve NaN propagation, comparisons retain
  NumPy-compatible ordered/unordered semantics, logical inputs normalize nonzero
  values to true, and bitwise operations preserve full integer bit patterns.
- **Positional operations:** RoPE supports mixed half-storage inputs with fp32
  angle tables; ALiBi supports generated or explicit per-head slopes.

### SSM, sparse attention, and EBM

- **Selective SSM:** forward and backward execute on gfx1151. Half-storage
  scalar-A workloads can use the chunked-parallel SSD path; f32 stays on the
  exact sequential scan to avoid the observed half-path overflow behavior.
- **MSA/DK2:** selected-block sparse attention and GPU-resident top-k execute,
  including GQA grouping, causal position masking, and dense-equivalence when
  all blocks are selected.
- **EBM:** native lanes cover Philox-driven Langevin steps, manifold samplers,
  exact-partition log-sum-exp, and fused EBT-tiny inference.

## Performance findings

Performance claims are scoped to the gfx1151 WSL system above. They are useful
for choosing the next compiler lever, not as portable expectations for discrete
RDNA or CDNA devices.

### GEMM ladder

| Rung | Finding | Decision |
|---|---|---|
| Register blocking | A 2×4 output macro-tile was about 2.3× the 1×1 baseline at 1024³/2048³. A 3×4 tile added roughly 20–25% at 3072³/4096³; 4×4 crossed the VGPR/occupancy cliff. | Production selection stays size-adaptive within the measured safe region. |
| LDS staging | Cooperative LDS staging lost at most tested sizes and gained only about 6% at 2048³. | Keep as a reference and as infrastructure for devices where global memory is the bottleneck. |
| Two-stage LDS pipeline | Double buffering gained about 8% only in a narrow 1024³–2048³ window and regressed outside it. | Do not promote as the general default. |
| Integer WMMA | At 2048³, f16/bf16/int8/int4 were all near the same matrix-op rate (roughly 21–24 T/op/s). | Treat int4 primarily as a footprint/bandwidth opportunity on gfx1151, not a compute-rate win. |

The next credible GEMM levers are VGPR-budgeted macro-tiling, occupancy, and
WMMA issue/scheduling. More staging is not the default answer on this APU's
unified LPDDR5x memory system.

### Flash-attention ladder

| Path | Landed improvement | Measured result and interpretation |
|---|---|---|
| Forward | Removed Q LDS staging, then fused online-softmax rescaling into P@V. | About 4.0→6.8 TFLOP/s at D=64 and 2.4→3.18 at D=128. LDS traffic and barriers, not only nominal occupancy, were limiting. |
| Backward | Replaced the scalar `_pre` work with WMMA and skipped provably empty causal tiles. | Roughly 3.4× for the WMMA rewrite and about 1.7× for causal tile skipping. |
| Further LDS reduction | Moving the large dK/dV accumulators to registers would exceed the VGPR budget and spill. | No clean incremental rung; meaningful gains require a different tiling or multi-wave design. |
| KV pipelining | Current kernels are WMMA/occupancy/LDS-bound rather than staging-bound. | Accepted-deferred until a staging-bound workload or architecture appears. |

### APU zero-copy

The opt-in `TESSERA_ROCM_ZEROCOPY=1` path maps host buffers directly on the
shared-memory APU. It changes end-to-end launch latency, not kernel throughput,
and falls back to the portable copy path if host registration is unavailable.

| GEMM size | Copy | Zero-copy | Result on WSL |
|---|---:|---:|---|
| 256³ | 0.54 ms | 2.40 ms | Copy wins |
| 512³ | 0.68 ms | 2.77 ms | Copy wins |
| 768³ | 5.44 ms | 3.52 ms | Zero-copy 1.5× |
| 1024³ | 7.17 ms | 4.15 ms | Zero-copy 1.7× |
| 1536³ | 10.45 ms | 8.47 ms | Zero-copy 1.2× |
| 2048³ | 13.86 ms | 10.13 ms | Zero-copy 1.4× |

Both registration and allocation involve Windows-driver round trips under WSL,
so the crossover is environment-specific. Zero-copy remains opt-in until
ROCM-8 supplies bare-metal evidence.

## Promotion policy

A ROCm feature is promoted only when all of the following refer to the same
exact target:

1. the compiler produces a target-valid artifact;
2. the runtime matrix contains an executable launch path;
3. a checked-in fixture compares device output with a numerical oracle;
4. the evidence records the device architecture that actually ran;
5. generated target-map and conformance views agree.

`device_verified_abi` additionally requires a shipped C-ABI symbol.
`device_verified_jit` identifies a compiler-generated device binary reached
through the runtime. Artifact emission without launch and numerical proof stays
`artifact_only`.

Performance promotion is separate from correctness promotion. A performance
change must be device-timed, compared with the current baseline, checked across
representative aligned and ragged shapes, and guarded by a ratchet where the
hardware lane is available.

## Source material

- [`STRIX_HALO_EXECUTION_PLAN.md`](STRIX_HALO_EXECUTION_PLAN.md) contains the
  detailed gfx1151 bring-up ladder and historical stage notes.
- [`ROCM_PATTERNS_FROM_AMD_ECOSYSTEM.md`](ROCM_PATTERNS_FROM_AMD_ECOSYSTEM.md)
  maps reusable ideas from AITER, ATOM, hipBLASLt, rocWMMA, Mori, Iris, XIO, and
  AMD's Gluon GEMM tutorial into Tessera work.
- [`../../../reference/isa/rdna/README.md`](../../../reference/isa/rdna/README.md)
  indexes the regenerable RDNA 3/3.5/4 ISA archive used for feature gating.
- [`../archive/nvidia_rocm_execute_and_compare_plan.md`](../archive/nvidia_rocm_execute_and_compare_plan.md)
  is the consolidated predecessor for the original cross-backend execution plan.
