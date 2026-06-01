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
  `test_apple_gpu_buffer_pool` assertion).
- **Done (M2/M3/M5):** the M2 scan (`Wh/Wx/xseq/init/ys`) and M3/M5 simdgroup
  matmul (`A/B/C`) large buffers now acquire through `TS_METAL_BUF_ACQUIRE*` too;
  the tiny dims/bias/params buffers stay fresh raw allocs (not worth pooling).

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
  0.61 ms → **0.28 ms** (~2.2×) on top of P1.
- **Done (residency-set reuse):** `MetalDeviceContext` now holds **one reusable
  `MTLResidencySet`**, repopulated per dispatch (`mtl4_set_residency`:
  removeAll + addAllocation + commit + requestResidency) and attached to the
  command buffer via **`[cb useResidencySet:res]`** inside `mtl4_encode_and_wait`
  — the granular intended path — instead of a fresh set + queue
  `addResidencySet:`/`removeResidencySet:` per dispatch. Wired into
  `mtl4_matmul2d_dispatch` (plain + epilogue), the conv lane (both stages share
  the one set), and the M8 session `run()`/`run_dev()` (W/bias/params stay
  resident via the session's persistent queue-level `resW`; only X/Y go in the
  per-step set). **Perf is neutral** (0.94–1.06×, in the noise — `commit` +
  `requestResidency` were the real per-call cost and are unavoidable), but it
  **eliminates the queue's 32-residency-set ceiling** (a latent footgun under
  many/racing dispatches) and matches the intended Metal 4 API. Kept for
  robustness, not speed. (The bespoke M2 scan / M3/M5 paths still create their own
  per-call set — cold paths, not worth converting.)

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

**Native `mpp::tensor_ops::convolution2d` — multi-tile CRACKED (2026-05-31).**
The previously-blocked multi-tile spike now works end-to-end via
**grid-of-threadgroups + per-tile slice on both source and destination**:
`tessera_apple_gpu_spike_conv2d_{single,multi}_tile_f16` in
`apple_gpu_runtime.mm`. Bit-correct vs numpy (rel ≤ 3e-7) at (8×8) / (16×16) /
(24×16) / (32×32). The key insight (not in the headers): the cooperative op
expects bound tensors whose **dimensions match the descriptor's source/dest**,
so you `X.slice(...)` + `Y.slice(...)` per-threadgroup with innermost-first
offsets (matching the matmul2d pattern); `set_offsets((K-1)/2)` only handles
the SAME→VALID halo, **not** the per-tile shift. Two additional findings the
spike surfaced:
- `MTLTensorDescriptor` requires **innermost-first dims with `strides[0]==1`**;
  the first try with numpy NHWC outermost-first order failed with
  `"Tensor Descriptor Validation: Stride (256) at index 0 should be 1"`. Fixed
  in `make_buffer_tensor_4d`.
- The conv2d header comment says "full threadgroup" scope but only
  `execution_thread` and `execution_simdgroups<N>` compile; `<4>` (128 threads)
  works (same as matmul2d).

**A/B vs MPSGraph fused conv (Cin=Cout=4, K=3, f16 in / f32 out, Apple M*):**
| dst H×W | native µs | MPSGraph µs | speedup |
|---------|-----------|-------------|---------|
|   8×8   | 361       | 406         | 1.12×   |
|  16×16  | 347       | 409         | 1.18×   |
|  32×32  | 355       | 460         | 1.30×   |
|  64×64  | 357       | 506         | 1.42×   |
| 128×128 | 377       | 774         | 2.05×   |
| 256×256 | 507       | 1183        | 2.33×   |

The native path wins at every shape and the gap widens with size (memory traffic
matters — no col materialization). **But before declaring a default-on lane**,
the spike's scope is narrow: hardcoded `Cin=Cout=4`, `K=3`, stride=1, no
dilation/groups, aligned-only tile (dstH/dstW multiples of 8), f16 only. Also
the MPSGraph baseline writes f16 output while the native path writes f32 — the
2× memory-traffic difference *favors* MPSGraph in this bench (so the native win
is conservatively understated, or the comparison is mixed-precision-skewed
depending on viewpoint). The honest disposition:
- **Land** the two spike entry points (`spike_conv2d_{single,multi}_tile_f16`)
  as opt-in C ABI symbols so the proof is preserved + regression-locked.
- **Don't flip** routing — productionizing this lane needs Cin/Cout templating
  (the descriptor's channel count is `constexpr`), non-aligned tile boundary
  handling, K/stride/dilation/groups generalization, and bf16 + f16-output
  epilogue parity with MPSGraph. Those are tracked as follow-ups; the existing
  im2col+matmul2d lane continues to deliver correct arbitrary-size conv today.

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
  ops* (zero-copy chaining), not for one isolated session step.

  **Resident MLP chaining (built, measured — capability yes, speedup no).** Added
  the two pieces for round-trip-free stacking: a **resident f32↔f16/bf16 cast**
  (`DeviceTensor.cast_to`, an elementwise MSL kernel — the M8 `run_dev` outputs f32,
  the next layer wants f16) and a **general both-operands-resident matmul**
  (`apple_gpu_matmul2d_dev` / `tessera_apple_gpu_mtl4_matmul2d_dev` — the complement
  to the fixed-weight session, for e.g. resident `Q @ Kᵀ`). A 3-layer MLP now runs
  entirely on the GPU (`run_dev` → resident `cast_to(f16)` → next layer), **bit-exact
  vs numpy**. **But it is *slower*, not faster** — 0.49–0.81× the host
  round-trip path (`run()` + numpy cast per layer), even with all intermediates
  preallocated (so it's the extra per-layer cast **dispatch+sync**, not buffer
  churn). The unified-memory verdict from the single-op bridge holds at chain
  scale: a host round-trip is a cheap memcpy, and any added GPU dispatch costs more.
  So these ship as **correct, tested capabilities** (resident dtype conversion;
  both-resident matmul) — *not* wired as an auto fast-path. A real win would need a
  **fused f16-output epilogue** (session writes f16 directly, no separate cast
  dispatch), and even then likely marginal given the memcpy is cheap.

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

**f16 / bf16 (landed).** MPS decomposition is f32-only, so a `_linalg_dtype_policy`
at the Python boundary runs f16/bf16 inputs on the GPU **in f32 and casts the
result back** (the bf16-matmul pattern); f32 is native; **f64 routes to numpy in
full precision** (it previously silently downcast to f32 — fixed). Covers
cholesky / solve / cholesky_solve / tri_solve.

**QR (landed — Cholesky-QR, GPU, verified).** No MPS QR kernel, so
`apple_gpu_qr` reuses the lane: `G = AᵀA`, `R = chol(G)ᵀ` (upper, +diagonal),
`Q = A·R⁻¹` via one lower-triangular solve — all on the existing GPU Cholesky +
tri-solve kernels, no new MSL. Cholesky-QR loses ~κ(A)² accuracy, so the result
is **verified** (`‖QᵀQ − I‖`); if it fails tolerance (or the Gram isn't PD, or
dtype is f64) it **falls back to numpy Householder QR**, so the returned `Q` is
always orthonormal. Tall/square (m ≥ n), reduced mode. Validated by
reconstruction + orthonormality (QR is unique only up to column signs).

**SVD (landed — one-sided Jacobi MSL).** MPS ships no SVD/eigensolver, so
`apple_gpu_svd` is a **custom MSL kernel**: one threadgroup per matrix, T=128
threads cooperating; it rotates pairs of *columns* of a working copy by
Givens/Jacobi rotations (accumulated into V) until every pair is mutually
orthogonal (a sweep with no rotation = converged), then σₖ = ‖col‖, U = normalized
columns, V = accumulated rotations. The m-dim dot products (α,β,γ) reduce through a
threadgroup tree; the sweep+pair loops run inside the kernel. The Python wrapper
sorts σ descending, builds `Vh = Vᵀ`, and **verifies `‖U·Σ·Vh − A‖`** before
trusting the iterative result — falling back to numpy on failure or for
`full_matrices=True` / f64. f16/bf16 compute in f32. Validated vs numpy:
reconstruction + U/V orthonormality ~1e-6, σ-match ~1e-6, and exact on
rank-deficient (tail σ = 0) and clustered/repeated σ.

**Batched + wide + Brent–Luk (landed).**
- **Batched (`…,m,n`)** runs **one threadgroup per matrix in a single grid
  dispatch** (`threadgroup_position_in_grid` indexes the slice) — whole-GPU
  utilization, measured **~30–95× a per-matrix dispatch loop** (the win grows with
  batch). Multi-leading-dim batches (e.g. `[2,3,20,5]`) flatten to one batch axis.
- **Wide (`m < n`)** runs `SVD(Aᵀ)` (tall → the kernel handles it) with U/V
  swapped — pure Python, no kernel change. Reduced dims preserved.
- **Brent–Luk parallel tournament** is now the **default** for `N ≤ 256`: the N/2
  disjoint column pairs of each round rotate concurrently — one per SIMD-group,
  each reducing its pair via `simd_sum` (no threadgroup scratch, no intra-round
  barrier; barrier only between rounds). **Measured 1.9–3.8× the sequential cyclic
  kernel** and numerically identical; the sequential batched kernel remains the
  `N > 256` fallback. (Contrast the SIMD-reduction rowop attempt, which regressed
  and was reverted — this one wins, so it ships.)

**Remaining follow-ups.** Native **batched** decomposition/solve (Cholesky/LU) if
MPS ever exposes a batch stride; on-GPU `full_matrices`. Honest perf note: the
*Cholesky/LU/tri-solve* batched per-matrix loop is correctness/capability-first —
each slice is a full GPU encode+commit+wait, so for many small matrices it is
**not** a speed win over numpy's batched LAPACK; the value is keeping the work
on-GPU when it's part of a larger resident pipeline.

## GPU-native RNG lane (opt-in, 2026)

`MPSMatrixRandomPhilox`-backed uniform / normal f32 fills:
`tessera_apple_gpu_random_{uniform,normal}_f32` (Python
`apple_gpu_random_uniform` / `apple_gpu_random_normal` → `(array, ran_on_gpu)`).
Philox-family (matching Tessera's S4 RNG family) and deterministic by `seed`, but
the stream is **not** bit-identical to Tessera's CPU Philox — so it is a
**separate opt-in surface, deliberately not wired into the deterministic
`tessera.rng` samplers** (that would break CPU/GPU equality + `check_determinism`,
Decision #18). Use it for large on-device random fills where generating on the CPU
and uploading would dominate. Validated by distribution (range/mean/std) + seed
determinism; guarded by `tests/unit/test_apple_gpu_rng.py`.

Implementation notes: the fill encodes into a flat **`MPSVector`**
(`encode:destinationVector:`) — the idiomatic 1-D path, no 2-D `rowBytes`
alignment to reason about. `MPSMatrixRandomPhilox` is **Philox-4x32** (generates 4
values per counter step) and asserts the element count is a multiple of 4, so the
runtime over-allocates to the next multiple of 4 and copies back the requested `n`
— odd lengths are handled (regression-locked). `batchStart`/`batchSize` on
`MPSMatrixRandom` are *batch-indexing* (which sub-arrays of a batched destination
to fill), not a Philox counter-offset for stream continuation, so they add nothing
to the single flat-fill surface and are intentionally unused.

## MPS device-caps / PackedFloat3 / MPSMatrixRandom — assessment (2026)

Quick verdicts on three further MPS surfaces, checked against the Tahoe SDK:

- **`MPSDeviceSupportsSimdReduction` / `SimdShuffle` / `SimdShuffleAndFill` /
  `SimdgroupBarrier`** — these are **Swift-only `MPSDeviceCaps`** (not in the C
  headers we compile) and are **unconditionally true on every Apple GPU that runs
  MSL 4.0** (M-series). **Shipped anyway as honest introspection:**
  `tessera_apple_gpu_simd_caps()` (Python `apple_gpu_simd_caps()`) queries the
  active device via `MTLGPUFamily` and returns a 4-bit mask
  (reduction / shuffle / shuffle-and-fill / simdgroup-barrier). On this M-series
  Mac it reports `0xF` (all four). Guarded by `test_apple_gpu_metal4.py`.

  **SIMD-reduction rowop optimization — tried, measured slower, reverted.** The
  one place a SIMD-group reduction could replace a threadgroup-memory tree is the
  tiled fused `matmul_softmax_tiled_{f32,f16}` kernels (`T == 32 == one
  SIMD-group`, so `simd_max`/`simd_sum` collapse the ~10-barrier tg_max/tg_sum
  tree to two ops). Implemented it, verified bit-correct, and **A/B-benchmarked vs
  the tree across M=128–256, N=1024–2048: 0.72–0.93× (slower).** These kernels are
  **matmul-K-loop + global-write bound, not reduction-bound**, and `simd_sum`/
  `simd_max` (which lower to a 5-deep shuffle dependency chain) cost more than the
  on-chip 32-lane tree here. Reverted the kernels (with an in-code note); kept the
  caps probe. Same discipline as the ThunderMittens result: the "obvious"
  optimization regressed, so it doesn't ship.
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
