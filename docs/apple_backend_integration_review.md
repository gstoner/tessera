# Apple Backend integration review (Metal 4 grounded)

> Reviewed: May 2026, against macOS 26.5 (Tahoe) SDK headers
> (`Metal.framework`, `MetalPerformancePrimitives.framework`) and the runtime in
> `src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm`.
> Purpose: confirm the Apple GPU integration is using the device the way the
> Metal 4 / MPS / MPSGraph APIs intend, and enumerate the gaps to "most optimal."

## Method

Read the authoritative SDK Objective-C headers (`MTLTensor.h`, `MTLBuffer.h`,
`MTL4ArgumentTable.h`, `MTL4ComputeCommandEncoder.h`, `MTLResidencySet.h`,
`MTL4CommandAllocator.h`, `MTL4CommandQueue.h`) and the MPP MSL headers
(`MPPTensorOpsMatMul2d.h` + impl), then cross-checked the runtime's three
execution lanes:
- **MPSGraph lane** (Tier-1 activations/norms, fused chains) — classic
  `MTLCommandQueue`/`MTLCommandBuffer`.
- **MSL lane** (hand-written kernels: matmul/softmax/gelu/rope/flash_attn/…) —
  classic command model, `compile_msl_kernel` cache.
- **MTL4 lane** (M2–M8: cooperative-matrix + MPP `matmul2d` + sessions) — the
  Metal 4 command model.

## What is already optimal (keep)

1. **Device + classic command queue are process singletons** (`deviceContext()`,
   `std::call_once`). No per-call device/queue creation. ✔
2. **Buffer pool** (`metal_buffer_acquire`/`_release`, bucketed 16 B–4 MB, RAII
   `TS_METAL_BUF_ACQUIRE*` macros) — recycles `MTLResourceStorageModeShared`
   buffers, saving the ~50–100 µs/alloc the profiling note documents. Used at
   **211 call sites** across the MPSGraph + MSL lanes. ✔
3. **MPSGraph graphs are cached** by (shape-class, opcode, dtype, shape[, eps,
   weighted]) (`mpsg_graph_cache`); MSL pipelines cached by (source, entry); MTL4
   pipelines + the MTL4 compiler + the MTL4 queue cached on the context. The
   ~1 ms compile/build costs are paid once. ✔
4. **Unified-memory exploited** — shared-storage buffers + `[buf contents]` give
   zero-copy host↔GPU; `TsDeviceTensor` (R0) keeps activations resident so a
   producer op can feed a consumer without a host round-trip. ✔
5. **M8 resident-weight session** — for decode, weights/pipeline/residency/queue
   are reused across steps; measured 3.3–3.6× faster per step than per-call. ✔

## Findings (prioritized)

### P1 — the MTL4 lane bypassed the shared buffer pool *(partially fixed here)*

Every MTL4 dispatch (M2 scan, M3/M5 matmul, M6/M7 epilogue, M8 session run,
spec-accept) allocated its buffers with raw `[device newBufferWithBytes:]` /
`newBufferWithLength:` — **0 of the 211 pool sites were on the MTL4 lane.** That
is 3–5 fresh allocations per dispatch, the exact churn the pool was built to
remove. Because the MTL4 one-shot paths **sync before returning** (they
`waitUntilSignaledValue:` then `memcpy`), the buffers are safe to recycle on
scope exit.

- **Done:** `mtl4_matmul2d_dispatch` (plain + epilogue, f16/bf16 — 4 entry
  points) now acquires A/B/C through `TS_METAL_BUF_ACQUIRE*`. Correctness
  unchanged; recycles on repeated same-size calls (buffers > 4 MB still bypass
  the pool and allocate fresh, which is correct).
- **Done (extended):** the M8 session `run()` X/Y buffers and `msl_spec_accept`
  now also use the pool (the latter fixed a pre-existing red
  `test_apple_gpu_buffer_pool` assertion). **Follow-up:** the M2 scan and M3/M5
  simdgroup matmul still allocate raw; the tiny bias/params buffers are not worth
  pooling.

### P2 — per-dispatch MTL4 object churn (argument table, allocator, command buffer) *(done)*

Each MTL4 dispatch used to create a **new** `MTL4ArgumentTable`,
`MTL4CommandAllocator`, and `MTL4CommandBuffer`. The Metal 4 design intends these
to be reused: `MTL4CommandAllocator` has `reset`, and an `MTL4CommandBuffer` "can
be reused immediately after committing."

- **Done:** `MetalDeviceContext` now holds a reusable allocator + command buffer
  + argument table (`maxBufferBindCount = 8`, covers all current kernels), reset
  + rebound each dispatch via `mtl4_encode_and_wait`. A dedicated
  `mtl4_dispatch_mu` serializes the encode→commit→wait sequence, which makes the
  reuse correct (and loses no overlap — the single shared queue already serializes
  GPU work). Wired into `mtl4_matmul2d_dispatch` (plain + epilogue, f16/bf16) and
  the M8 session `run()`. **Measured:** repeated small epilogue (64×256×256) went
  0.61 ms → **0.28 ms** (~2.2×) on top of P1. The per-call `MTLResidencySet` is
  still created fresh (commit + `requestResidency` are unavoidable kernel calls);
  reusing the set object is a minor remaining nicety. **Header-check note:**
  `MTL4CommandBuffer` exposes `useResidencySet:` / `useResidencySets:count:` —
  per-command-buffer residency that is the more granular intended path than our
  queue-level `addResidencySet:`/`removeResidencySet:` churn (and sidesteps the
  queue's 32-residency-set ceiling). `[cb useResidencySet:res]` with one reused,
  repopulated set is the clean form of the residual above.

### P3 — `MTLSharedEvent` created per dispatch *(done)*

Every MTL4 dispatch used to `[dev newSharedEvent]` then signal value 1. Apple's
"Running an ML model on the GPU timeline" sample treats the event as a reusable
resource advanced with `signaledValue + 1`.

- **Done:** one `MTLSharedEvent` per device, advanced by a monotonic counter
  (`mtl4_event_val`) in `mtl4_encode_and_wait`. Correct under the `mtl4_dispatch_mu`
  serialization (no out-of-order signals). Folded into the P2 helper.

### P4 — MTL4 binary archive (pipeline persistence) *(done, opt-in)*

MTL4 pipelines were recompiled on every fresh process start (~ms each).

- **Done:** `tessera_apple_gpu_mtl4_archive_enable(path)` loads a prior archive
  (`newArchiveWithURL:`) as a lookup archive (`MTL4CompilerTaskOptions.lookupArchives`,
  so matching pipelines skip the MSL recompile) and attaches a `CaptureBinaries`
  `MTL4PipelineDataSetSerializer` to the compiler; `..._archive_flush()` writes the
  captured set (`serializeAsArchiveAndFlushToURL:`). Python:
  `runtime.apple_gpu_mtl4_archive_enable(path)` / `apple_gpu_mtl4_archive_flush()`.
  **Opt-in** (off by default — no effect on the default path) because capture only
  covers pipelines built *after* enable, so it must be called at init before the
  first MTL4 op. Verified by a fresh-process round-trip test (process 1 writes a
  ~50 KB archive; process 2 loads it and stays correct). Degrades cleanly off
  Metal 4 (returns False).

### P5 — bf16 matmul routes to the native tensor-op by default *(done)*

The legacy Apple GPU bf16 matmul (`tessera_apple_gpu_mps_matmul_bf16`) converts
to fp32 on the host because **MPS has no native bf16 GEMM**.

- **Done:** `_apple_gpu_dispatch_matmul` now routes rank-2 bf16 matmul to
  `tessera_apple_gpu_mtl4_matmul2d_bf16` **by default** when Metal 4 is available
  (`_mtl4_route_matmul2d_bf16`), casting the f32 accumulator back to bf16 to
  preserve the bf16-in/bf16-out contract. Unlike the f32 lane (opt-in), bf16 is
  default-ON because there is no good MPS bf16 path to regress against. Toggle via
  `TESSERA_APPLE_GPU_MTL4_BF16=0` / `set_apple_gpu_mtl4_bf16_default(False)`.
  **Measured end-to-end (`@jit(target="apple_gpu")` bf16 matmul):** MTL4-default
  vs forced-legacy = **14.7× (1024³), 11.8× (2048³)** faster. First default
  routing flip onto the MTL4 lane. (f32 remains opt-in: the MPS f32 GEMM is
  well-tuned and the hand kernel only reaches ~80% of it.)

### P6 — compile-time `linear+bias+activation` fusion *(done)*

- **Done:** `driver._apple_gpu_chain_kind` recognizes `matmul/gemm → add(bias)
  [→ gelu|relu|silu]` structurally (the trace leaves operand dtypes as `?`, so the
  dtype decision moves to runtime), and `runtime._execute_apple_gpu_mps_metadata`
  dispatches it via `_apple_gpu_dispatch_matmul_bias_act`: **f16/bf16 with a
  per-column [N] bias → one MPP `matmul2d` epilogue dispatch** (bias + act fused
  in-register, fp32-accumulated — so the result is *more* accurate than the per-op
  f16 chain); otherwise the matmul stays on MPS (GPU) with host bias+act. So
  `gelu(linear(x, W, b))` under `@jit(target="apple_gpu")` now auto-fuses, and even
  f32 / residual-add cases get the matmul on-GPU (a win over the all-numpy eager
  path these multi-op chains hit before). Tested by
  `test_p6_linear_bias_act_fuses_to_epilogue` (f16/bf16 × none/gelu/relu/silu) +
  `test_p6_residual_add_falls_back_correctly`.

### P7 — `MTLTensorUsageMachineLearning` path unused

All MTL4 tensors are created with `MTLTensorUsageCompute`. The
`MTL4MachineLearningCommandEncoder` + `MTLTensorUsageMachineLearning` path
(running a compiled `.mtlpackage`) is a different, higher-level surface Tessera
does not use. This is **correct for now** — hand-written cooperative kernels give
the fusion control MPSGraph/CoreML cannot — but worth revisiting if Tessera ever
wants to run whole compiled subgraphs on the ML encoder.

### P8 — conv on the matrix units *(done, opt-in; native conv op cracked single-tile)*

**Shipped:** an f16/bf16 conv lane on the GPU matrix units via **im2col + the M7
`matmul2d` epilogue** — a KxK conv equals `im2col(activation) @ weights_reshaped`,
and conv's per-output-channel bias is exactly the epilogue's per-column bias, so
`conv → bias → activation` collapses to one fused matmul2d dispatch (fp32-
accumulated, any size, stride/padding/dilation, groups=1). `runtime.apple_gpu_conv2d(...)`
+ wired into `_apple_gpu_dispatch_conv2d`. Correct to ≤3e-2 vs a dtype-matched
reference across f16/bf16 × {none,relu,gelu,silu} × stride/pad/dilation × 1×1
(`tests/unit/test_apple_gpu_metal4.py::test_p8_conv2d_matmul2d_lane`).

**GPU im2col (landed).** The unfold now runs **on the GPU** (an MSL `im2col`
kernel; `apple_gpu_conv2d` prefers the on-device `tessera_apple_gpu_mtl4_conv2d_*`
symbol — `col` never leaves the GPU — with host im2col only as a fallback). Two
MTL4 dispatches (im2col → matmul2d epilogue) in the reusable-object path; correct
to ≤2e-5 across stride/padding/dilation/1×1 (a partial-M-tile grid-swap bug was
found + fixed — the matmul2d kernel slices A-rows by `tg.x`, so the conv grid must
be `(M_tiles, N_tiles)`).

**Still opt-in, OFF by default.** On-device im2col removed the host-gather cost
(host ~0.5× → GPU ~0.55–0.77× of MPSGraph) but **still loses to MPSGraph's *fused*
conv** (f16 0.65–0.77×; bf16 0.54–0.57× vs the MPSGraph-f32 legacy path).
Materializing the `col` matrix is extra memory traffic a fused/direct/Winograd
conv avoids — so unlike P5 matmul, conv has **no easy default-win via im2col**.
The path to beating MPSGraph is the native MPP `convolution2d` cooperative op (no
col materialization — multi-tile tiling undocumented) or a direct conv, not
materialized im2col. Toggle `TESSERA_APPLE_GPU_MTL4_CONV=1`.

**Native `mpp::tensor_ops::convolution2d` — investigated, conventions cracked,
multi-tile blocked.** The cooperative conv op compiles and is **bit-correct
single-tile** (VALID 3×3 ≤2e-7). Reverse-engineered conventions: NHWC activation
/ HWIO weights / NHWO dest tensor extents (innermost-first), the op does a
**SAME/centered window** so `set_offsets((K-1)/2,(K-1)/2)` yields VALID conv,
float `cooperative_tensor` destination + epilogue (same as matmul2d), full-
threadgroup scope, **compile-time descriptor dims**. The blocker is **multi-tile
grid-tiling**: both slice-based (matmul2d-style) and offset-based tiling produce
wrong results, and there is no usage example in the headers. So the native conv
op is a future swap-in for the im2col lane's matmul core once Apple documents (or
we crack) its tiling; the im2col+matmul2d lane delivers correct arbitrary-size
conv today.

## ThunderMittens (Metal port of ThunderKittens) — reviewed; key technique does NOT transfer

Reviewed HazyResearch's [ThunderMittens](https://github.com/HazyResearch/ThunderMittens)
(MSL port of ThunderKittens) and the [blog](https://hazyresearch.stanford.edu/blog/2024-11-28-tk-mlx)
for matmul/attention speedups. Its `kernels/matmul_custom/matmul_custom.metal`
GEMM (11 lines, "~9% faster than MLX" on M2 Pro) uses **no threadgroup/shared
staging** — it loads global→register directly, register-blocks with 8×8 base
tiles (`metal::simdgroup_float8x8`), full-unrolls K, and skips double-buffering;
the blog's thesis is *"shared memory isn't as crucial… keep ALUs active by loading
HBM→registers, almost never leveraging shared memory for reuse."*

**Tested on this Mac (GPU-timestamped f32, direct global→register, register-blocked
32×32 and 64×64): it is SLOWER, not faster** — ~3.5 / 4.3 TFLOP/s vs the M5
threadgroup-**staged** kernel's ~6.6 and MPS's ~8.0. The technique is M2-Pro-tuned
(~200 GB/s, ~6.5 TFLOP/s f32 peak); on this newer/higher-bandwidth Apple GPU,
staging a K-slab once into fast on-chip memory and reusing it across all
simdgroups beats re-reading from device per simdgroup. So M5's staging instinct is
correct here — **do not adopt the no-shared-memory approach.** (TM also benchmarks
against MLX, which is weaker than MPS; "9% faster than MLX" ≠ faster than MPS.)

**What does transfer (and we'd already arrived at):** occupancy-first design,
≤8 `simdgroup_float8x8` accumulators/thread to avoid register spill, padding (not
swizzling) for bank conflicts, and 8×8 (not 16×16) base tiles. **What TM predates:**
the MSL 4.0 MetalPerformancePrimitives `matmul2d` cooperative op (M6) — TM uses raw
`simdgroup_multiply_accumulate`; our `matmul2d` path *beats* MPS for fp16, so for
half precision the cooperative op is the better lever than TM's hand-rolled
simdgroup GEMM. TM's clean tile DSL (`rt`/`st`/`gl` types, warp/group ops) is a
nice model for a future Tile-IR→MSL lowering, but not a perf win.

## R-series device-resident lane — Metal 4 review (2026)

Re-examined R0–R4 (the GPU-resident-activation surface) against the better Metal 4
documentation to see whether any should adopt the MTL4 command model / typed
`MTLTensor` resources.

- **R0 (`TsDeviceTensor`)** — a shared `MTLBuffer` + nbytes. *The one genuine
  Metal 4 gap.* It fed the MPSGraph lane (R1/R2/R4) fine, but **could not feed the
  MTL4 cooperative (matrix-unit) lane** without a host round-trip — every MTL4 path
  re-uploaded host arrays via `newBufferWithBytes`. **Fixed (R0→MTLTensor bridge):**
  `TsDeviceTensor` now carries a lazily-created, cached **buffer-backed `MTLTensor`
  view** (`ts_dev_tensor_view`, cached by `(inner, outer, dt)`), and a new
  `tessera_apple_gpu_mtl4_mlp_session_run_dev(handle, X, Y, M)` binds resident X/Y
  tensor views straight into the M8 session's argument table — **no X upload, no Y
  download**. `AppleGPUMLPSession.run_dev(X_devtensor[, Y_devtensor])` exposes it;
  bit-exact vs `run()`/reference (rel 0.0 f16/bf16), guarded by
  `test_apple_gpu_metal4.py::test_mtl4_mlp_session_run_dev_*` (4 tests).

  **Honest perf note:** at decode sizes the per-step saving is **within noise**
  (0.96–1.13× across M=1..32, K=4096) — on unified memory the saved memcpy (X ≈ 8 KB,
  Y ≈ 16–44 KB) is negligible against the ~1–2 ms dispatch+compute. The bridge's
  value is **architectural**: it's the missing capability that lets a resident
  activation reach the matrix-unit lane at all without a round-trip. The latency
  win materializes only when it removes a *download→reupload between successive MTL4
  ops* (zero-copy chaining), not for one isolated session step. Full round-trip-free
  MLP stacking additionally needs a resident f32→f16 cast between layers (follow-up).

- **R1 (`bmm_dev`), R2 (`TsEncodeSession`: bmm/unary/binary/gather/rowop/gumbel),
  R4 (block-paged gather)** — all **MPSGraph-based on the classic command model**,
  consuming R0 buffers zero-copy *within the MPSGraph lane*. **They cannot move to
  the MTL4 command model:** MPSGraph has zero MTL4 support (it only encodes to a
  classic `MTLCommandBuffer`), so a port would mean abandoning MPSGraph. They are
  already optimal for that lane — R2's "encode N ops into one command buffer, commit
  once" is the right MPSGraph amortization. Their MTL4 analogue already exists
  *separately* as the M8 resident-weight session + P2/P3 reusable-object dispatch.
  **Verdict: leave R1/R2/R4 as-is.**

- **R3** — unassigned; the R-series numbering skips it.

## GPU linear-algebra lane — Cholesky / LU / triangular solve (2026)

**Shipped.** A dense f32 linear-algebra lane on the GPU via the
MetalPerformanceShaders `MPSMatrix*` fixed-function kernels —
`MPSMatrixDecompositionCholesky`, `MPSMatrixDecompositionLU`,
`MPSMatrixSolveCholesky`, `MPSMatrixSolveLU`, `MPSMatrixSolveTriangular`. This is
the **one capability MPSGraph cannot provide** — it has no matrix-decomposition
ops — so before this lane `tessera.ops.{cholesky, solve, cholesky_solve,
tri_solve}` had **no GPU path at all** (numpy/CPU reference only), despite having
VJPs registered. It's the single place the legacy `MPSMatrix*` family offers
something nothing else in the Apple stack can.

Runtime C ABI (rank-2 f32): `tessera_apple_gpu_cholesky_f32`,
`tessera_apple_gpu_solve_cholesky_f32`, `tessera_apple_gpu_solve_lu_f32`,
`tessera_apple_gpu_tri_solve_f32`. Each returns `0` on a successful GPU run, `2`
for a singular / non-positive-definite matrix, `-1` if Metal is unavailable — so
the Python wrapper cleanly falls back to numpy (which raises the same
`LinAlgError` a pure-numpy call would). Python:
`runtime.apple_gpu_{cholesky, solve, cholesky_solve, tri_solve}(...)`, each
returning `(result, ran_on_gpu)`.

- **Matrices are row-major** (MPSMatrix native) — matches numpy storage, so **no
  transpose at the boundary** (verified bit-correct first try). Cholesky uses
  `lower:YES` and zeroes the strict-upper triangle to match
  `numpy.linalg.cholesky`; the solves chain decomposition → solve internally (LU
  pivots fed straight from `MPSMatrixDecompositionLU` into `MPSMatrixSolveLU`).
- **Triangular solve** reads only the relevant triangle (BLAS trsm semantics),
  matching `np.tril`/`np.triu`; `lower`/`trans`/`unit` flags all supported.
- **Correctness:** rel ≤ 2e-4 vs an f64 numpy reference across Cholesky (+
  reconstruction), SPD solve, general LU solve (matrix + vector RHS), and
  triangular solve over the full `{lower,upper}×{trans}×{unit}` matrix. Uses the
  RAII buffer pool (`TS_METAL_BUF_ACQUIRE*`) like every other dispatcher. Guarded
  by `tests/unit/test_apple_gpu_linalg.py` (24 tests).

**Batched + `@jit` wiring (landed).** **Batched (`ndim>2`) f32 now runs on the
GPU** — the Python wrappers loop the rank-2 kernel per matrix (MPS
decomposition/solve are single-matrix per encode — there is no native batch
API), pairing batched RHS slices for the solves; matrix and stacked-vector RHS
both supported, numpy fallback if any slice doesn't run. **`@jit(target="apple_gpu")`
dispatch is wired** for the two registered Graph IR ops — `tessera.cholesky` and
`tessera.tri_solve` (honoring the `lower` attribute) — via `_APPLE_GPU_LINALG_OPS`
in both `driver.py` (envelope/gate) and `runtime.py` (`_apple_gpu_dispatch_linalg`),
so `ts.ops.cholesky(A)` / `ts.ops.tri_solve(A, b)` in a jitted apple_gpu function
report `execution_mode="metal_runtime"` on Darwin and compute on the GPU
(verified bit-correct vs numpy). The full-solve functions (`apple_gpu_solve`,
`apple_gpu_cholesky_solve`) remain direct `runtime.*` utilities — there is no
`tessera.solve` Graph IR op to route through `@jit`.

**Remaining follow-ups.** **f16** (MPS decomposition is f32-only — would need a
different route); native **batched** (one dispatch for the whole batch) if MPS
ever exposes a batch stride for decomposition; **QR/SVD** have no MPS kernel at
all (custom MSL or a different route). Honest perf note: the batched per-matrix
loop is correctness/capability-first — each slice is a full GPU
encode+commit+wait, so for many small matrices it is **not** a speed win over
numpy's batched LAPACK; the value is keeping the work on-GPU when it's part of a
larger resident pipeline.

## MPS device-caps / PackedFloat3 / MPSMatrixRandom — assessment (2026)

Quick verdicts on three further MPS surfaces, checked against the Tahoe SDK:

- **`MPSDeviceSupportsSimdReduction` / `SimdShuffle` / `SimdShuffleAndFill` /
  `SimdgroupBarrier`** — these capability flags are **Swift-only `MPSDeviceCaps`
  values** (not in the public C/Obj-C headers we compile against) and, more to the
  point, are **unconditionally true on every Apple GPU that runs MSL 4.0 / Metal
  4** (M-series, A11+). Querying them in the backend buys nothing — there's no
  target in our envelope that lacks them. The *intrinsics themselves*
  (`simd_sum`/`simd_max`/`simd_shuffle`) **are** a real lever: a SIMD-group row
  reduction skips the threadgroup-memory round-trip our softmax/rmsnorm/layernorm
  MSL kernels currently use. So the optimization worth pursuing is using the simd
  reduction intrinsics in those rowop kernels — **not** wiring the cap queries.
- **`MPSPackedFloat3`** — a 12-byte packed 3-vector for **ray-tracing / geometry
  acceleration structures** (`MPSRayIntersector` et al.). It is **not a tensor-math
  type** and has no role in a DL/HPC compiler; using it "for math ops" would be a
  category error. Skip.
- **`MPSMatrixRandomPhilox` / `MPSMatrixRandomMTGP32`** — **genuinely relevant** to
  the S4 RNG lane (`tessera.rng` is Philox-backed), and a real GPU capability
  (fill an MPSMatrix/MPSVector with uniform/normal on-device). **But** MPS's
  Philox-4x32 stream will **not** be bit-identical to Tessera's CPU Philox
  reference — different counter/key layout — so it **cannot transparently
  accelerate the existing deterministic samplers** without breaking the
  CPU/GPU-equality + `check_determinism` contracts (Decisions #18). It's viable
  only as a **separate GPU-native RNG path** whose determinism is defined by the
  MPS generator's own seed. Worth it for large on-device `randn`/`rand` fills
  where the host→device copy of CPU-generated noise dominates; lower priority than
  the linalg/decode work, and gated on a clear "GPU-RNG stream ≠ CPU-RNG stream"
  contract decision.

## Verification against the Metal 4 SDK headers (2026 review)

The API claims in this doc and in `apple_gpu_metal4_adoption.md` were
cross-checked against the Tahoe (macOS 26.5) SDK headers. **Confirmed correct:**
- The MTL4 command-model usage — `MTL4Compiler
  newComputePipelineStateWithDescriptor:compilerTaskOptions:error:`, argument-table
  binding (`setResource:atBufferIndex:` with `tensor.gpuResourceID`,
  `setAddress:atIndex:`), buffer-backed `MTLTensor`
  (`newTensorWithDescriptor:offset:error:`), `MTLSharedEvent` sync.
- The MPP `matmul2d` cooperative pattern — `tensor_handle` `MTLTensor` operands
  (not in-kernel `tensor_inline`), float `cooperative_tensor` accumulator,
  `is_valid_element`/`get_capacity`/`get_multidimensional_index`/`store`; and that
  **f32 has no `execution_simdgroups` path** (matrix units are fp16/bf16).
- P4 archive flow — `MTL4PipelineDataSetSerializerConfigurationCaptureBinaries`,
  `serializeAsArchiveAndFlushToURL:`, `newArchiveWithURL:`,
  `MTL4CompilerTaskOptions.lookupArchives`.
- The "MPSGraph runs only on the classic `MTLCommandQueue`, never an MTL4 queue"
  claim — there are **zero** `MTL4` references in the MPSGraph framework headers,
  so the additive-lane premise holds.

**Gaps the header re-check surfaced:** P8 (`convolution2d` cooperative op unused —
material), and the residency refinement (`MTL4CommandBuffer useResidencySet:`,
noted in P2). The cooperative-tensor `reduce_rows`/`reduce_columns` ops (in the
matmul2d header) are also unused — a future in-register softmax/attention-reduction
fusion opportunity.

## Status

- **Done:** P1 (buffer pool on matmul2d + session + spec_accept), P2 + P3
  (reuse allocator / command buffer / argument table / shared event, serialized),
  P4 (opt-in `MTL4Archive` pipeline persistence), P5 (default bf16 → native MTL4,
  **11.8–14.7×** the legacy fallback), P6 (compile-time `linear+bias+act` →
  `matmul2d` epilogue auto-fusion under `@jit`).
- **Done:** **P8 — f16/bf16 conv on the matrix units** via im2col + the M7
  matmul2d epilogue (correct, any size, fused bias/act; opt-in pending a GPU
  im2col). Native `convolution2d` cooperative op cracked single-tile.
- **Remaining:** P8 perf — **GPU im2col** so the gather stays on-device (then the
  conv lane flips on like P5); the native `convolution2d` multi-tile tiling
  (undocumented). Plus minor polish: P1 tail (pool the M2 scan + M3/M5 simdgroup
  matmul), the `useResidencySet:` residency refinement (P2), unused cooperative
  reductions.

Net: every API claim in both docs was header-verified and is correct. All of
Metal 4's core ML compute capabilities are now exercised: the matmul family is
fully exploited (fp16 beats MPS, bf16 beats the conversion fallback ~10–15×,
fused epilogue, resident-weight session, default bf16 routing, pipeline archives),
and conv runs on the matrix units (P8, correct + fused, opt-in until GPU im2col).
The `MTL4MachineLearningCommandEncoder` (compiled-model inference) is deliberately
not used — the right call for a compiler that emits its own fused kernels.
