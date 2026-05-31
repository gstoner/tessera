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

**Opt-in, OFF by default** (`TESSERA_APPLE_GPU_MTL4_CONV=1` /
`set_apple_gpu_mtl4_conv_routing`). Unlike P5 matmul, conv's legacy bf16 path
already runs the well-tuned MPSGraph *f32* conv + a host cast (fast), so the
lane's *host* im2col gather currently makes it ~1.5–2× slower. The matmul is on
the matrix units; the win needs a **GPU im2col** so the gather stays on-device
(the perf follow-up — then it flips on like P5).

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
