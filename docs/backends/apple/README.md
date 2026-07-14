# Apple backend — CPU + GPU compiler support (reference)

> **This is the canonical architecture and implementation reference for Tessera's
> Apple M-series backend** — CPU (Phase 8.2, Accelerate/BNNS/LAPACK) and GPU
> (Phases 8.3 → 8.4.8, the Metal 4 lane, and the `runtime.launch()` op-parity
> lanes). It is the single narrative doc for the Apple backend; the design
> plans and surveys that fed it (Metal 4 adoption ladder, control-flow
> lowering, resident-activations plan, Tier-2/3 plan, capability roadmap, MLX
> ecosystem survey) have all landed and are archived under
> [`docs/audit/backend/apple/archive/`](../../audit/backend/apple/archive/); the old
> `docs/*.md` stub paths have been removed. See the **Archived documents**
> section at the bottom for the full index + links.
>
> **The kernel + lane inventory is its own doc:**
> [kernel-guide.md](kernel-guide.md) — the native
> MSL/MPS/MPSGraph/linalg/RNG symbol tables **plus** the `runtime.launch()`
> reference/compute lanes that reach op-family parity with x86/ROCm.
>
> Related: [GA / EBM status](../../status/ga_ebm.md).
>
> **Execution status is generated, not maintained here.** Read
> [`apple_execution_inventory.md`](../../audit/generated/apple_execution_inventory.md)
> for the execution-unit taxonomy; [`apple_target_map.md`](../../audit/generated/apple_target_map.md)
> for generic per-op dispatch; and
> [`runtime_execution_matrix.md`](../../audit/generated/runtime_execution_matrix.md)
> for observed executor placement. These distinguish generic JIT dispatch,
> strict Value Target-IR calls, package-authored subgraphs, and CPU-reference
> fallback.

---

## Status at a glance

- **Apple CPU** — the **generic** `@jit(target="apple_cpu")` route is
  Accelerate/BNNS GEMM (rank-2 f32/f16/bf16; rank-3 f32 batched) plus explicit
  NumPy fallback for other ops. Separately, the strict Value Target-IR lane has
  native single-call LAPACK execution for Cholesky, triangular/Cholesky solve,
  LU, QR, and SVD. Do not conflate that narrow executable allowlist with
  generic multi-op CPU codegen.
- **Apple GPU (Phases 8.3 → 8.4.8)** — `@jit(target="apple_gpu")` via MPS +
  custom MSL kernels: 9 kernel concepts × {f32, f16, bf16} = 26 core runtime
  symbols, 4 fused chains, threadgroup-tiled large-N `matmul_softmax`.
- **MPSGraph lane (2026-05-29)** — Tier-1 activations/norms + a broad long
  tail through Apple's optimizing graph compiler; no `N ≤ 256` limit;
  completes the f16/bf16 + large-N fused-chain matrix.
- **Metal 4 lane (M0–M8 + P-series)** — MTL4 command model + `MTLTensor` +
  MetalPerformancePrimitives cooperative `matmul2d`: an evidence-gated fp16/bf16
  matmul and cheap-epilogue lane, plus resident-weight MLP session and opt-in
  matrix-unit convolution. Selection is per device, deployment target, shape,
  dtype, and fused region; a Metal 4 feature alone does not promote a route.
- **GA / EBM (2026-05-18)** — 17/17 GA + 9/9 native EBM primitives ship fused
  MSL kernels; `@clifford_jit(target="apple_gpu")` lowers AST →
  `CliffordIRProgram` at decoration time.
- **GPU linear algebra** — dense f32 Cholesky / LU / triangular-solve / QR /
  SVD via `MPSMatrix*` + custom MSL Jacobi kernels (the one lane MPSGraph
  cannot supply).
- **Op-family parity (2026-07-10)** — beyond the perf kernels above, Apple GPU
  has `runtime.launch()` lanes for the same op families the x86/ROCm device
  lanes cover: conv, losses, complex/geometric + conformal, Philox RNG, linalg
  (lu/qr/svd/cholesky-solve) + einsum/factorized, optimizers, reductions +
  0-move/sort, scatter, sparse/MoE, MLA latent-KV, and speculative-decode. Each
  is F4-gated against the standalone reference and reported honestly per lane —
  `native_gpu` where it genuinely dispatches to MPS/MSL (loss/complex/reduce,
  numpy fallback off-Metal), `reference_cpu` where Apple ships no device kernel.
  The only remaining gap is `quantize`/`dequantize` fp4/fp6/fp8/nvfp4. It needs
  the macOS 27 / Metal 4.1 tensor toolchain *and* device-specific numerical and
  throughput evidence; availability alone is not a native-performance claim. Full table:
  [kernel-guide.md](kernel-guide.md) §2.

f32 is fully wired for the named native lanes. f16/bf16 are wired for matmul,
rope, softmax, gelu, flash_attn, and the fusion chains; individual execution
rows remain the authoritative dtype and placement record.

---

## Architecture — IR flow + execution gates

```
@jit(target="apple_gpu")  /  @jit(target="apple_cpu")
        │
        ▼
Graph IR  ──►  Schedule IR  ──►  Tile IR  ──►  Target IR (apple_cpu | apple_gpu)
                                                  │
                                                  ▼
                 tessera-lower-to-apple_{cpu,gpu}-runtime  (or artifact-only path)
                                                  │
                          ┌───────────────────────┴───────────────────────┐
                          ▼                                                 ▼
            CPU: generic GEMM or value-call ABI          GPU: MPS/MSL/MTL4 or package-subgraph ABI
                                                                          │
                                                                          ▼
                                      classic compute encoder  /  MTL4 ML package encoder
                                                                          │
                                                                          ▼
                                                                on-device execution
```

Three layered decisions gate **GPU** execution:

1. **Compile-time gate** (`driver._is_apple_gpu_mps_executable`) — does the
   program qualify for runtime execution at all? Yes →
   `execution_mode = "metal_runtime"`; no → `metal_artifact`.
2. **Chain detection** (`driver._apple_gpu_chain_kind`) — for multi-op
   programs, which fusion pattern (if any) does the SSA chain match?
3. **Runtime dtype dispatch** (`runtime._apple_gpu_dispatch_*`) — at JIT call
   time, the input array dtype picks the f32 / f16 / bf16 ctypes wrapper,
   since `@jit` is type-polymorphic.

### Compile-time vs runtime dispatch (important nuance)

`@jit(target="apple_gpu")` is **type-polymorphic** — the decorator does not
see call-site dtypes. So:

- The static **Graph IR** assumes `f32` operands unless explicit type hints
  are present.
- The compile-time **backend artifact** names `tessera_apple_gpu_*_f32`
  symbols by default.
- At **launch time** the `_apple_gpu_dispatch_*` helpers inspect the
  `numpy.dtype` of the actual inputs and route to the matching ctypes wrapper.

This split is why **lit fixtures** test compile-time dtype selection with
explicit `tensor<*xf16>` types, while **Python unit tests** test runtime
dtype dispatch with `np.float16` / `ml_dtypes.bfloat16` arrays. The two
paths share one ABI; they only differ in *when* the symbol is chosen.

---

## Apple CPU backend

The generic `@jit(target="apple_cpu")` route executes its GEMM operations
natively through an Accelerate / BNNS shim. The lowering pass `MatmulToAppleCPU`
(`lib/Target/Apple/Lowering/MatmulToAppleCPU.cpp`) turns static-shape rank-2
f32 `tessera.matmul` into `func.call @tessera_apple_cpu_gemm_f32`; the
pipeline alias is `tessera-lower-to-apple_cpu-runtime` (parallel to the
artifact-only `tessera-lower-to-apple_cpu`).

Runtime shim `runtime/apple_cpu_runtime.cpp` (built as `TesseraAppleRuntime`,
links `-framework Accelerate` on Darwin, portable reference fallback
elsewhere):

| Symbol | Path | Notes |
|--------|------|-------|
| `tessera_apple_cpu_gemm_f32` | Accelerate | single rank-2 f32 GEMM via `cblas_sgemm` |
| `tessera_apple_cpu_gemm_f32_batched` | Accelerate | rank-3 batched GEMM looping `cblas_sgemm` per batch |
| `tessera_apple_cpu_gemm_f16` | BNNS | rank-2 fp16 GEMM via `BNNSMatMul` (native fp16, cblas+fp32 fallback) |
| `tessera_apple_cpu_gemm_bf16` | BNNS | rank-2 bf16 via `BNNSDataTypeBFloat16` (macOS 12+), bit-shift fp32 fallback |

Python boundary:

- `_execute_apple_cpu_accelerate_artifact` chains arbitrary supported op
  sequences: matmul/gemm dispatch to Accelerate, every other supported op
  falls through to the NumPy reference. Multi-op programs are first-class, but
  this is **not** a claim of native codegen for their non-GEMM operations.
- `_apple_cpu_dispatch_matmul` selects rank-2 f32 / rank-3 batched f32 /
  rank-2 fp16 (BNNS) / `np.matmul` fallback.
- bf16 uses `ml_dtypes.bfloat16` (a soft import — the bf16 fast path is
  unavailable when `ml_dtypes` is absent, the rest of the runtime keeps
  working).
- **Launch-overhead reduction:** `runtime_artifact()` is lazily cached;
  `apple_cpu` `__call__` bypasses `runtime.launch()` via `_apple_cpu_fast_call`.

**Measured speedups (Apple Silicon, Accelerate active):** 8³ GEMM 459 µs →
10 µs (**46×**); 32³ 456 µs → 12 µs (**38×**); 128³ 470 µs → 19 µs (**25×**);
512³ 780 µs → 193 µs (**4×**). At 512³ Tessera launch overhead is ~1.3× numpy
(was ~5×).

End-to-end verified on this Mac (LLVM/MLIR 22.1.6, Accelerate active): single
GEMM bitwise-matches numpy; multi-op tiny decode bitwise-matches the numpy
reference; rank-3 batched GEMM bitwise-matches numpy; fp16 matmul matches an
f32-converted reference at fp16 tolerance. Tests:
`tests/unit/test_apple_backend_roadmap.py`.

### Strict Apple Value Target-IR lane

The `tessera-lower-to-apple_cpu-runtime-full` path is a distinct, value-preserving
execution unit. It lowers one supported operation to `tessera_apple.cpu.call`,
carries the named C ABI symbol in artifact metadata, and executes only the
allowlisted symbol—never by a parallel op-name fallback. Alongside GEMM, this
currently covers f32 Cholesky, triangular solve, Cholesky solve, LU, QR, and
SVD through Accelerate/LAPACK. It is deliberately restricted to a **single
value call** per artifact today; composing several calls is future compiler
work, not evidence that the generic multi-op JIT is native.

The generated [Apple execution inventory](../../audit/generated/apple_execution_inventory.md)
is the canonical status view for this split. The Value Target-IR proof surface
is `tests/unit/test_apple_value_target_ir.py`.

---

## Apple GPU execution lanes

The GPU backend has several coexisting lanes, picked per op/chain:

| Lane | Command model | Used for |
|------|---------------|----------|
| **MSL** | classic `MTLCommandQueue` + `compile_msl_kernel` cache | hand-written kernels: matmul/softmax/gelu/rope/flash_attn + fused chains |
| **MPS** | classic | `MPSMatrixMultiplication` for rank-2 matmul |
| **MPSGraph** | classic | Tier-1 activations/norms + long tail + f16/bf16/large-N fused-chain epilogues |
| **MTL4** | Metal 4 command model | cooperative `matmul2d` (fp16/bf16), fused epilogue, resident-weight session |
| **MPSMatrix** | classic | dense f32 linear algebra (Cholesky/LU/tri-solve) — the lane MPSGraph can't supply |
| **RNG** (opt-in) | classic | `MPSMatrixRandomPhilox` uniform/normal fills |

### Pipeline ordering

`tessera-lower-to-apple_gpu-runtime` runs passes longest-fusion-first so the
most specific match wins greedy pattern matching:

```
1. matmul_softmax_matmul fusion   (3 ops, benefit=3)
2. matmul_softmax       fusion    (2 ops, benefit=2)
3. matmul_gelu          fusion    (2 ops, benefit=2)
4. matmul_rmsnorm       fusion    (2 ops, benefit=2)
5. matmul (mps)         lowering  (single op)
6. rope                 lowering
7. flash_attn           lowering
8. softmax              lowering
9. gelu                 lowering
(+ Tier-1 MPSGraph passes: tessera-unary-to-apple_gpu /
   tessera-silu-mul-to-apple_gpu / tessera-rowop-to-apple_gpu)
```

A `matmul → softmax → matmul` chain would otherwise be caught by the 2-op
`matmul → softmax` pass first, losing the 3-op opportunity — hence the
ordering.

---

## GPU kernel inventory

> The full kernel + lane inventory now lives in its own doc:
> **[kernel-guide.md](kernel-guide.md)** — the native
> MSL / MPS / MPSGraph / linear-algebra / RNG symbol tables (single-op, fused
> 2-op/3-op, the MPSGraph long tail, the `MPSMatrix*` factorizations,
> capability/diagnostic symbols, the ABI summary, and the core-9 coverage
> matrix) **plus** the `runtime.launch()` reference/compute lanes that reach
> op-family parity with x86/ROCm (§2). Machine-readable truth:
> [`runtime_abi.csv`](../../audit/generated/runtime_abi.csv).

The **GPU linear-algebra implementation state** (batched / QR / SVD details,
perf, and `@jit` wiring) is documented in the Metal 4 section below.
## Apple Silicon deep-learning datatypes

> Reference note. Apple does not publish a raw per-chip Neural Engine datatype
> ISA table. This tracks the datatypes Apple exposes through public frameworks
> (Core ML, Metal, MPS, Accelerate/BNNS, MLX).

| Datatype | CPU | GPU | Apple NPU / Neural Engine |
| --- | --- | --- | --- |
| FP64 | Yes (MLX: CPU-only) | Not exposed for MLX GPU execution | No public ANE exposure |
| FP32 | Yes | Yes | Core ML may accept FP32; ANE placement is runtime-dependent |
| FP16 | Yes | Yes (preferred for many workloads) | Yes, commonly used via Core ML |
| BF16 | Yes (MLX + framework APIs) | Yes (MLX, Metal, MPS) | Not publicly documented as a raw ANE datatype |
| INT8 / UINT8 | Yes | Yes | Yes via Core ML quantized inference |
| INT4 / UINT4 | Mainly quantized storage/weights | Yes (Metal tensor datatypes + quantized ML) | Not directly documented |
| FP8 | Not generally exposed | Not in the public Metal tensor datatype list | Not publicly exposed |
| MXFP4 | MLX M5 examples | MLX M5 benchmark examples | Not a public ANE datatype claim |

**Per-generation notes:** M1 (16-core NE, 11 TOPS) through M4 (16-core NE,
38 TOPS) expose the same FP32/FP16/BF16 + INT8/INT4 software contract via
Core ML / Metal / MPS / MLX. **M5** is the first generation where Apple
publicly describes **GPU Neural Accelerators in each GPU core** with direct
developer access through **Metal 4 Tensor APIs** (and MLX benchmarks BF16 /
4-bit / MXFP4 workloads).

**Tessera lowering strategy (conservative):**

1. Treat `f32`, `f16`, `bf16` as the main floating-point GPU-visible datatypes.
2. Treat `f64` as CPU-only unless a runtime path proves otherwise.
3. Model `int8`/`uint8`/`int4`/`uint4` primarily as inference / quantized-weight datatypes.
4. Keep Neural Engine lowering behind Core ML / framework-mediated execution rather than assuming direct ANE control.
5. For M5+, consider a separate GPU TensorOps / Neural-Accelerator capability bit rather than folding it into the generic Apple GPU profile.

*"Exposed" means available through public Apple APIs, not necessarily executed
natively by every hardware block for every op.* Sources: Apple Developer docs
(Core ML compute devices; Metal `MTLTensorDataType`; MPS `MPSDataType`), MLX
data-types docs, and Apple Newsroom M1–M5 announcements.

---

## Metal 4 lane — implementation state

> The Metal 4 *ladder* (M0–M8 + P-series — what to build, in what order) has all
> landed; it is archived at
> [`docs/audit/backend/apple/archive/apple_gpu_metal4_adoption.md`](../../audit/backend/apple/archive/apple_gpu_metal4_adoption.md).
> This section is the *review of what has actually landed*, cross-checked against
> the macOS 26.5 (Tahoe) SDK headers (`MTLTensor.h`, `MTL4*.h`,
> `MTLResidencySet.h`, the MPP `matmul2d` headers) and the runtime in
> `apple_gpu_runtime.mm`.

### What is already optimal (keep)

1. **Device + classic command queue are process singletons**
   (`deviceContext()`, `std::call_once`) — no per-call device/queue creation.
2. **Buffer pool** (`metal_buffer_acquire`/`_release`, bucketed 16 B–4 MB,
   RAII `TS_METAL_BUF_ACQUIRE*`) recycles `MTLResourceStorageModeShared`
   buffers (saves ~50–100 µs/alloc); used at **211 call sites** across the
   MPSGraph + MSL lanes.
3. **Graphs/pipelines cached** — MPSGraph by `(shape-class, opcode, dtype,
   shape[, eps, weighted])`; MSL pipelines by `(source, entry)`; MTL4
   pipelines + compiler + queue on the context.
4. **Unified memory exploited** — shared-storage buffers + `[buf contents]`
   give zero-copy host↔GPU; `TsDeviceTensor` keeps activations resident.
5. **M8 resident-weight session** — weights/pipeline/residency/queue reused
   across decode steps; measured 3.3–3.6× faster per step than per-call.

### Findings (P1–P8) — status

| # | Finding | Status |
|---|---------|--------|
| **P1** | MTL4 lane bypassed the shared buffer pool | **Done** — matmul2d (plain + epilogue, f16/bf16), M8 session run, spec-accept, M2 scan + M3/M5 simdgroup matmul large buffers now acquire via `TS_METAL_BUF_ACQUIRE*` |
| **P2** | per-dispatch MTL4 object churn (argument table / allocator / command buffer) | **Done** — reusable allocator + command buffer + argument table, serialized by `mtl4_dispatch_mu`; one reusable `MTLResidencySet` via `[cb useResidencySet:]`. Repeated 64×256×256 epilogue 0.61 → **0.28 ms** (~2.2×) on top of P1 |
| **P3** | `MTLSharedEvent` created per dispatch | **Done** — one event/device, advanced by a monotonic counter |
| **P4** | MTL4 binary archive (pipeline persistence) | **Done, opt-in** — `tessera_apple_gpu_mtl4_archive_enable(path)` / `_flush()`; fresh-process round-trip verified |
| **P5** | bf16 matmul routes to native tensor-op by default | **Done** — rank-2 bf16 → `tessera_apple_gpu_mtl4_matmul2d_bf16` by default; **14.7× (1024³), 11.8× (2048³)** vs forced-legacy. Toggle `TESSERA_APPLE_GPU_MTL4_BF16=0`. (f32 stays opt-in — MPS f32 GEMM is well-tuned) |
| **P6** | compile-time `linear+bias+activation` fusion | **Done** — `matmul/gemm → add(bias) [→ gelu|relu|silu]` auto-fuses to one MPP `matmul2d` epilogue (fp32-accumulated); f32/residual cases still get the matmul on-GPU |
| **P7** | ML encoder not selected by generic JIT | **Open (intentional)** — `.mtlpackage` author/load/dispatch exists, but `MTL4MachineLearningCommandEncoder` is not yet selected as a generic `package_call` route. Keep hand-written cooperative kernels for individual fused regions; revisit only for resident, whole compiled subgraphs. |
| **P8** | conv on the matrix units | **Done, opt-in** — f16/bf16 conv via im2col + M7 matmul2d epilogue (GPU im2col landed); native `mpp::tensor_ops::convolution2d` multi-tile **cracked** (`spike_conv2d_{single,multi}_tile_f16`, bit-correct). Still off by default — im2col loses to MPSGraph's *fused* conv; native spike is narrow (Cin=Cout=4, K=3, f16). Toggle `TESSERA_APPLE_GPU_MTL4_CONV=1` |

### Packaged ML subgraphs

The `.mtlpackage` runtime lifecycle—package authoring, load/compile,
reflection, tensor preparation, ML-encoder dispatch, and numerical proof—is
implemented. It is **not** the default individual-kernel path: MPS/MSL/MPP
remain preferable when Tessera needs exact fusion control. The deferred
compiler integration is selection and execution of a package-backed *whole
subgraph*: `package_call` must carry a binding spec, package identity, shapes,
and resident-tensor ABI, then be selected only when it beats the ordinary
MPSGraph or fused-MSL route. The generated execution inventory records this as
`package_call launch gated`, rather than calling package authoring generic JIT
execution.

### GPU linear-algebra implementation state

The dense f32 lane (Cholesky / LU / triangular-solve, symbols in the inventory
above) is the one capability MPSGraph cannot provide — before it,
`tessera.ops.{cholesky, solve, cholesky_solve, tri_solve}` had no GPU path at
all despite registered VJPs.

- **Correctness** rel ≤ 2e-4 vs an f64 numpy reference across Cholesky (+
  reconstruction), SPD solve, general LU solve (matrix + vector RHS), and the
  full `{lower,upper}×{trans}×{unit}` triangular matrix. Row-major (no boundary
  transpose). Uses the RAII buffer pool.
- **Batched (`ndim>2`)** loops the rank-2 kernel per matrix (MPS decomp/solve
  are single-matrix per encode). **Custom batched MSL grid kernels**
  (`cholesky_batched`, `tri_solve_batched` — one threadgroup per matrix) measured
  **40–388×** a per-matrix MPS loop; cholesky-solve = batched chol + 2 batched
  tri-solves. Batched general `solve` (LU) still loops the rank-2 MPS kernel.
- **`@jit(target="apple_gpu")` wiring** for the two registered Graph IR ops
  `tessera.cholesky` + `tessera.tri_solve` (via `_APPLE_GPU_LINALG_OPS` in
  `driver.py` + `runtime.py::_apple_gpu_dispatch_linalg`) → `metal_runtime` on
  Darwin. **f16/bf16** compute in f32 and cast back; **f64** routes to numpy.
- **QR** — Cholesky-QR (`G=AᵀA`, `R=chol(G)ᵀ`, `Q=A·R⁻¹`) reusing the GPU
  chol+tri-solve, with a `‖QᵀQ−I‖` verify → numpy-Householder fallback.
- **SVD** — custom one-sided **Jacobi MSL** kernels:
  `_svd_bl_batched_f32` (**Brent–Luk parallel tournament**, default N≤256,
  ~1.9–3.8× the cyclic kernel) / `_svd_batched_f32` (sequential, N>256) /
  `_svd_f32` (rank-2). Batched = one threadgroup per matrix per grid dispatch
  (~30–95× a loop); wide `m<n` via `SVD(Aᵀ)`; host sort + `‖UΣVh−A‖` verify →
  numpy (full_matrices / f64). Tests: `tests/unit/test_apple_gpu_linalg.py` (60).

### R-series device-resident lane — verdicts

- **R0 (`TsDeviceTensor`)** — the one genuine Metal 4 gap, now fixed: carries a
  cached buffer-backed `MTLTensor` view so resident activations reach the MTL4
  cooperative lane without a host round-trip
  (`mtl4_mlp_session_run_dev` binds resident X/Y). **Honest perf note:** on
  unified memory the saved memcpy is within noise at decode sizes; the value is
  *architectural* (zero-copy chaining), not raw latency. Resident dtype cast +
  both-resident matmul ship as **correct, tested capabilities, not auto
  fast-paths** (chaining measured 0.49–0.81× the host round-trip — extra
  per-layer dispatch costs more than a cheap memcpy).
- **R1/R2/R4** — MPSGraph-based on the classic command model; MPSGraph has zero
  MTL4 support, so they cannot move to the MTL4 model. Already optimal for that
  lane. **Leave as-is.** (R3 is unassigned.)

### Reviews that did *not* change the design

- **ThunderMittens** (MSL port of ThunderKittens) — its no-shared-memory
  global→register GEMM is M2-Pro-tuned; **measured slower** here (~3.5–4.3 vs
  the staged kernel's ~6.6 and MPS's ~8.0 TFLOP/s). Do not adopt. What transfers
  (occupancy-first, ≤8 `simdgroup_float8x8` accumulators, padding not swizzling,
  8×8 base tiles) we'd already arrived at; the MSL 4.0 `matmul2d` cooperative op
  is the better lever for half precision.
- **SIMD-reduction rowop** — replacing the threadgroup tree with
  `simd_max`/`simd_sum` in the tiled fused kernels **regressed** (0.72–0.93×):
  these kernels are matmul-K-loop + global-write bound, not reduction-bound.
  Reverted; kept the `simd_caps` probe.
- **`MPSPackedFloat3`** — a ray-tracing/geometry type, not tensor math. Skip.

### SDK verification

Every API claim here and in the adoption ladder was cross-checked against the
Tahoe SDK headers and confirmed: the MTL4 command-model usage, the MPP
`matmul2d` cooperative pattern (incl. that **f32 has no `execution_simdgroups`
path** — matrix units are fp16/bf16), the P4 archive flow, and the "MPSGraph
runs only on the classic queue, never an MTL4 queue" premise (zero MTL4
references in the MPSGraph headers). Gaps the re-check surfaced: P8
(`convolution2d` cooperative op) and the cooperative-tensor
`reduce_rows`/`reduce_columns` ops (an unused in-register
softmax/attention-reduction fusion opportunity).

---

## Anatomy of a kernel emission

Every apple_gpu MSL kernel has 4 cross-layer artifacts:

1. **MSL source string + cache key** — a top-level constant in
   `python/tessera/compiler/target_ir.py`
   (`_APPLE_GPU_FOO_MSL_SOURCE` + `_APPLE_GPU_FOO_MSL_CACHE_KEY =
   _sha256_short(...)`). The runtime keys its `MTLComputePipelineState` cache by
   `(msl_source, entry_point)`.
2. **C ABI symbol in `apple_gpu_runtime.mm`** — one function per (kernel ×
   dtype), e.g. `extern "C" void tessera_apple_gpu_foo_f32(...)`. Darwin →
   Metal/MPS; non-Darwin → portable C++ reference in `apple_gpu_runtime_stub.cpp`.
3. **Lowering pass in `lib/Target/Apple/Lowering/`** — a `RewritePattern`
   matching the Graph IR op (or chain) that emits a `func.call` into the runtime
   symbol (extract ptrs via `bufferization.to_buffer` +
   `memref.extract_aligned_pointer_as_index` + `arith.index_cast`, allocate
   output, emit shape/scale/eps constants, `ensureExternalDecl`, emit
   `func::CallOp`, `bufferization.to_tensor`, replace/erase).
4. **Python dispatcher + ctypes wrapper in `runtime.py`** — gate on
   dtype/shape, lazily load the symbol, set argtypes/restype, call. Invoked by
   `_execute_apple_gpu_mps_metadata` based on the metadata `op_name` (single-op)
   or chain detection.

### How to add a new single-op kernel

1. MSL source + C symbol in `apple_gpu_runtime.mm` + stub fallback
2. Lowering pass in `lib/Target/Apple/Lowering/FooToAppleGPU.cpp`
3. Pass declaration in `Passes.h`
4. Wire into the pipeline in `Passes.cpp` (per-op section, after fusions)
5. CMake source entry in `CMakeLists.txt`
6. MSL source constant + cache key in `target_ir.py` (extend
   `_apple_gpu_kernel_msl_for_dtype` for dtype variants)
7. Single-op envelope entry in `target_ir.py::_apple_gpu_module_is_mps_runtime`
   + `_lower_apple_gpu_op`
8. Driver gating — extend `driver.py::_APPLE_GPU_MSL_OPS` + artifact symbol selection
9. Python dispatcher + ctypes wrapper in `runtime.py` (envelope + dispatcher
   table + loader gate)
10. Tests — lit fixture + unit tests (artifact contract + end-to-end + ABI shim)

### How to add a new fusion (2-op / 3-op)

1. Fused MSL kernel + C symbol in `apple_gpu_runtime.mm`
2. Lowering pass matching the chain at the consumer, walking up to verify
   `hasOneUse()` on intermediates
3. Wire into the pipeline **before** per-op passes (longer chains first)
4. Driver chain detection — extend `_apple_gpu_chain_kind`
5. Target IR fusion-kind detection — extend `_apple_gpu_module_fusion_kind`
   (classify by source-set + op order) + per-pass MSL source/cache_key dispatch
6. Runtime metadata-chain detector — extend `_apple_gpu_metadata_is_*_chain`,
   add dispatcher entry
7. `_apple_gpu_module_is_mps_runtime` — accept the new chain shape (enforce chain
   ORDER, not just shape)
8. Tests following the existing pattern

---

## MetalDeviceContext + kernel cache

```cpp
struct MetalDeviceContext {
  id<MTLDevice>       device;
  id<MTLCommandQueue> queue;
  bool                ok;
  std::unordered_map<std::string, id<MTLComputePipelineState>> kernel_cache;
  std::mutex          kernel_cache_mu;
};
```

- **Process-wide singleton**, initialized lazily on first call.
- **Cache key** = `msl_source + '\x1f' + entry_point`.
- **Re-check under lock** in `compile_msl_kernel` for the concurrent-compile race.

The ~1 ms MSL compile cost is paid once per unique kernel per process;
subsequent dispatches are O(buffer setup + encode + commit + wait).

---

## Constraints summary

- **Per-thread stack-array kernels** (everything except `mps_matmul` and the
  MPSGraph lane): N ≤ 256 (or D ≤ 256 for flash_attn).
- **Threadgroup-tiled `matmul_softmax`** (Phase 8.4.6, f32 only): N ≤ 8192.
- **MPSGraph lane**: no N limit.
- **Static shapes only** — dynamic-shape paths fall back to artifact-only /
  numpy reference.
- **Single-use intermediates** in fusion patterns — multi-use falls back to
  per-op.
- **Matching dtypes within a chain** — mixed-dtype chains fall back to per-op.

---

## Test surface

- **Lit fixtures:** `tests/tessera-ir/phase8/apple_*.mlir` — compile-time
  symbol selection + pipeline composition (CPU + GPU; incl.
  `apple_gpu_tier1_lowering.mlir`).
- **Python unit tests:** `tests/unit/test_apple_backend_roadmap.py` (CPU +
  GPU end-to-end, runtime dtype dispatch, ABI shim, fusion gates, MSL cache),
  plus `test_apple_gpu_mpsgraph_lane.py`, `test_apple_gpu_llama_decoder_layer.py`,
  `test_apple_gpu_linalg.py`, `test_apple_gpu_metal4.py`, `test_apple_gpu_rng.py`,
  `test_apple_gpu_buffer_pool.py`, and the Tier-2/3 suites
  (`test_apple_gpu_bmm.py`, `_projections.py`, `_batched_mha.py`, `_reductions.py`,
  `_gqa.py`, `_fused_attention.py`).
- **Benchmark harness:** `benchmarks/apple_gpu/benchmark_fusion.py` (fused vs
  sequential) + `benchmark_ga_ebm.py`.
- **Generated machine-readable surface:** `docs/audit/generated/runtime_abi.csv`
  (drift-gated symbol inventory).

---

## Files at a glance

| Concern | Path |
|---------|------|
| CPU runtime shim (Accelerate/BNNS) | `src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_cpu_runtime.cpp` |
| GPU runtime shim (Darwin) | `src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm` |
| GPU runtime stub (non-Darwin) | `src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime_stub.cpp` |
| Lowering passes | `src/compiler/codegen/Tessera_Apple_Backend/lib/Target/Apple/Lowering/` |
| Pipeline registration | `.../lib/Target/Apple/Passes.cpp` |
| Pass declarations | `.../include/Tessera/Target/Apple/Passes.h` |
| MSL source constants + helper | `python/tessera/compiler/target_ir.py` |
| Compile-time gating | `python/tessera/compiler/driver.py` |
| Runtime dispatcher + ctypes | `python/tessera/runtime.py` |
| Benchmark harness | `benchmarks/apple_gpu/benchmark_fusion.py` |
| Unit tests | `tests/unit/test_apple_backend_roadmap.py` (+ the suites above) |
| Lit fixtures | `tests/tessera-ir/phase8/apple_*.mlir` |

---

## Apple GPU target descriptor — the contract boundary (Phase 3)

Apple GPU is modeled as an **explicit target descriptor**, not a bare
`target="apple_gpu"` string. The descriptor
(`python/tessera/compiler/apple_target_descriptor.py`) is the contract boundary
between *what the compiler produced* and *what the runtime can actually execute*.
It is emitted into the generated Target IR attrs
(`target_ir.lower_tile_to_target_ir` → `attrs["target_descriptor"]`) and carried
in `RuntimeArtifact.metadata["target_descriptor"]` (JSON round-trip preserved).

**Fixed identity:** `vendor="apple"`, `api="metal"`, `triple="air64-apple-macosx"`,
`arch="apple-metal"`, `memory_model="unified_64"`.

**Three DISTINCT execution contracts — never conflate them:**

| `execution_contract` | Meaning | Runtime claim |
|---|---|---|
| `metal_artifact` | compile-time artifact only | **none** — pure compile / containerization produces this |
| `metal_runtime` | classic MPS / MSL / MPSGraph lane (existing, mature) | runs on the classic command queue |
| `mtl4_runtime` | Metal 4 cooperative-tensor-op lane (MTL4 command model) | runs on `MTL4CommandQueue`/allocator/compiler + cooperative `tensor` ops |

**The lane split is load-bearing.** MPSGraph / classic-queue (`metal_runtime`)
is a *separate surface* from MTL4 cooperative tensor ops (`mtl4_runtime`).
MPSGraph does **not** run on the MTL4 command model; a module is never labeled
across the two. The MPS execution path in `apple_gpu_runtime.mm` is
`metal_runtime`, never `mtl4_runtime`.

**Metal 4 features are capability-gated requirements** (`required_capabilities`).
The vocabulary mirrors the runtime probe bits: `mtl4_command_queue`,
`mtl4_command_allocator`, `mtl4_compiler`, `mtl_tensor`, `msl_4_0`,
`mtl4_archive`, `mtl4_ml_encoder`. A `metal_artifact` / `metal_runtime` descriptor
requires none of the MTL4 gates; an `mtl4_runtime` descriptor requires the
command-model trio, and a cooperative-tensor op (e.g. MTL4 cooperative matmul)
additionally requires **`mtl_tensor` + `msl_4_0`**.

**Compile-time status is separate from observed runtime capabilities.** A
descriptor produced by pure compile carries `required_capabilities` (what it would
need) but never `observed_capabilities` (what a host has). Observed capabilities
come **only from a real runtime probe** —
`_apple_gpu_dispatch.apple_gpu_capabilities_snapshot()` is the canonical source,
surfaced by `runtime.query_backend("apple_gpu")["observed_capabilities"]`. When
the runtime can't load, the snapshot is explicit (`runtime_available=False`, empty
`capabilities`, `mtl4_full=False`) — no silent "Metal 4 full" claim.

**Explicit tensor / buffer ABI** (`apple_tensor_abi`): `dtype`, `rank`, `shape`,
`strides` (row-major default), `offset_bytes`, `resource_kind`
(`mtl_buffer` | `mtl_tensor_view` | `mtlpackage_tensor`), `resident` (true only
for device-resident / `TsDeviceTensor` paths), and the buffer-vs-tensor `view`.

**Contract guards** (`validate_descriptor` / `assert_not_artifact_claiming_runtime`):
an artifact-only module must not claim a runtime contract; a classic lane must not
smuggle MTL4 command-model caps; an `mtl4_runtime` descriptor's required caps are
checked against observed caps when a probe ran. Tests:
`tests/unit/test_apple_target_descriptor.py`.

---

## Archived documents

The plans, surveys, and reviews that used to sit alongside this file have been
folded into this reference (and the [kernel + lane
inventory](kernel-guide.md)). Their full historical text is
preserved for provenance under
[`docs/audit/backend/apple/archive/`](../../audit/backend/apple/archive/) — there are
no redirect stubs at the old `docs/` paths; link directly to the archive:

| Archived document | What it covered |
|---|---|
| [docs/audit/backend/apple/archive/apple_backend_capability_roadmap.md](../../audit/backend/apple/archive/apple_backend_capability_roadmap.md) | P1–P8 capability / performance ladder (packaged kernels, memory accounting, quant). |
| [docs/audit/backend/apple/archive/apple_gpu_metal4_adoption.md](../../audit/backend/apple/archive/apple_gpu_metal4_adoption.md) | The M0–M8 / R0 Metal 4 adoption ladder. |
| [docs/audit/backend/apple/archive/apple_gpu_control_flow_lowering.md](../../audit/backend/apple/archive/apple_gpu_control_flow_lowering.md) | Phase-G `if` / `while` → MSL control-flow mapping. |
| [docs/audit/backend/apple/archive/apple_gpu_resident_activations_plan.md](../../audit/backend/apple/archive/apple_gpu_resident_activations_plan.md) | Device-resident activation strategy. |
| [docs/audit/backend/apple/archive/apple_gpu_tier2_tier3_plan.md](../../audit/backend/apple/archive/apple_gpu_tier2_tier3_plan.md) | Tier-2 / Tier-3 packaged-kernel (`.mtlpackage`) roadmap. |
| [docs/audit/backend/apple/archive/apple_gpu_mlx_ecosystem_survey.md](../../audit/backend/apple/archive/apple_gpu_mlx_ecosystem_survey.md) | MLX reference-vocabulary survey (Decision #23 — reference only). |
| [docs/audit/backend/apple/archive/ldt_primitives_metal4_mapping.md](../../audit/backend/apple/archive/ldt_primitives_metal4_mapping.md) | Functional → perf path for `count_nonzero` / `popcount` / `asymmetric_bce` / `masked_categorical`. |

Older Apple provenance (chain audits, GA/EBM execution-gap notes, single-command-
buffer decode plan) also lives in that archive directory.
