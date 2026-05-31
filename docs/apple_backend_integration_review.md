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
- **Follow-up:** apply the same to the M8 session `run()` X/Y buffers, the M2
  scan, and the M3/M5 simdgroup matmul. The session's per-run X/Y are the
  highest-value remaining target (decode hot path). The tiny bias/params buffers
  are not worth pooling.

### P2 — per-dispatch MTL4 object churn (residency set, argument table, allocator, command buffer)

Each MTL4 dispatch creates a **new** `MTLResidencySet` (and calls `commit` +
`requestResidency`, which is a kernel transition), a **new** `MTL4ArgumentTable`,
a **new** `MTL4CommandAllocator`, and a **new** `MTL4CommandBuffer`. The Metal 4
design intends these to be **reused**: `MTL4CommandAllocator` has `reset`, the
docs state an `MTL4CommandBuffer` "can be reused immediately after committing,"
and a residency set is meant to be committed once and kept resident.

- **Recommendation:** add a small per-queue pool to `MetalDeviceContext`: one
  reusable command allocator + command buffer (reset/begin each dispatch), and a
  cached argument table per binding-arity (3 for plain, 5 for fused). The M8
  session already proves the residency-set-reuse win; generalize it. Expected to
  remove most of the small-size per-call overhead that keeps routing OFF.

### P3 — `MTLSharedEvent` created per dispatch

Every MTL4 dispatch does `id<MTLSharedEvent> ev = [dev newSharedEvent]` then
signals value 1. Apple's "Running an ML model on the GPU timeline" sample
explicitly treats the shared event as a **reusable resource** created once and
advanced with `signaledValue + 1`.

- **Recommendation:** cache one `MTLSharedEvent` per device and advance a
  monotonic value. **Caveat:** under concurrent submission the signal values must
  stay monotonic (you cannot signal a lower value after a higher one), so either
  serialize MTL4 submission behind `mtl4_mu` for the commit+signal pair, or keep
  per-call events when concurrency is expected. Low absolute cost; do it
  alongside P2.

### P4 — no MTL4 binary archive (pipelines recompiled each process)

MTL4 pipelines are cached in-process but recompiled on every fresh process start
(~ms each, growing with the kernel count: M2…M8 + the 4 `matmul2d` variants).
`MTL4Archive` / `MTL4BinaryFunction` persist compiled pipelines to disk.

- **Recommendation:** serialize the MTL4 pipeline set to an `MTL4Archive` keyed
  on (device, OS, source hash); load it at `deviceContext()` init. Meaningful for
  CLI / short-lived processes and for first-token latency.

### P5 — bf16 matmul still routes to an fp32-conversion fallback

The default Apple GPU bf16 matmul (`tessera_apple_gpu_mps_matmul_bf16`) converts
to fp32 on the host because **MPS has no native bf16 GEMM**. M6 added a native
bf16 `matmul2d` tensor-op that is ~10× that fallback.

- **Recommendation:** once P1+P2+P3 amortize the MTL4 per-call overhead, route
  `@jit(target="apple_gpu")` bf16 matmul to `tessera_apple_gpu_mtl4_matmul2d_bf16`
  by default (it is strictly better than the conversion path). This is the
  clearest case where the MTL4 lane should become the default for a dtype.

### P6 — fused `linear+bias+activation` not recognized at compile time

The fused epilogue kernel + session exist, but a user writing
`gelu(linear(x, W, b))` under `@jit(target="apple_gpu")` in f16/bf16 still lowers
through the per-op MPSGraph path unless the epilogue API is called explicitly.

- **Recommendation:** extend the existing `matmul→gelu` / `matmul→rmsnorm` Graph
  IR chain recognizer (`runtime.py` + `driver.py`) with a bias operand and an
  f16/bf16 gate so the chain collapses to one `matmul2d_epilogue` dispatch.

### P7 — `MTLTensorUsageMachineLearning` path unused

All MTL4 tensors are created with `MTLTensorUsageCompute`. The
`MTL4MachineLearningCommandEncoder` + `MTLTensorUsageMachineLearning` path
(running a compiled `.mtlpackage`) is a different, higher-level surface Tessera
does not use. This is **correct for now** — hand-written cooperative kernels give
the fusion control MPSGraph/CoreML cannot — but worth revisiting if Tessera ever
wants to run whole compiled subgraphs on the ML encoder.

## Recommended sequence

1. **P2 + P3** (reuse command allocator / buffer / argument table / shared event)
   — the largest remaining per-call overhead, and the prerequisite for any
   default routing flip.
2. **P1 follow-up** (pool the session run + scan + simdgroup matmul buffers).
3. **P5** (default bf16 → MTL4) and **P6** (compile-time epilogue fusion) — the
   two changes that turn the lane from "validated but off" into a real default
   win for half-precision transformer blocks.
4. **P4** (MTL4Archive) for process-start latency.

Net: the kernels are correct and competitive (fp16 matmul beats MPS, bf16 beats
the conversion fallback, the epilogue fuses for free); the gap to "most optimal"
is now almost entirely **per-call host-side overhead amortization**, which P1–P3
address directly and which the M8 session already validated as worth ~3.3×.
