# Metal 4 adoption — design + ladder

> Status: **M0 + M1 + M2 + M3 + M4 + M5 + M6 + M7 + M8 landed.** This machine runs macOS 26.5 (Tahoe) with
> SDK 26.5, so the full Metal 4 surface is available:
> `MTL4CommandQueue/Buffer/Allocator/Encoder`, `MTLTensor`, MSL 4.0
> (`MTLLanguageVersion4_0`) — all `API_AVAILABLE(macos(26.0))`. M2 lands the
> first real kernel through the full MTL4 command model, which doubles as the
> concrete **Phase-G → MSL 4.0** mapping (see
> docs/apple_gpu_control_flow_lowering.md).

## Why this is *additive*, not an upgrade

Tessera's Apple GPU backend is overwhelmingly **MPSGraph** (matmul, attention,
the Tier-1 lane, the Phase-G control-flow `forLoop`/`while` kernels, the R2
encode session). The public **MPSGraph headers do not expose MTL4
command-buffer encoding** — MPSGraph still runs on the *classic*
`MTLCommandQueue`/`MTLCommandBuffer`. So Metal 4 is **not** a drop-in upgrade to
the MPSGraph lane: there is no "run my existing MPSGraph on an MTL4 queue" path.

Adopting Metal 4 therefore means a **parallel backend lane** — kernels written
against the MTL4 command model + MSL 4.0 tensor ops + `MTLTensor` — running
*alongside* MPSGraph, capability-gated, with MPSGraph as the default/fallback.
MPSGraph is fully supported on Tahoe and is **not** deprecated; nothing here is
forced.

## What Metal 4 offers Tessera

| Metal 4 surface | What it buys | Relevance |
|---|---|---|
| `MTL4CommandQueue` + allocator/encoder model | Lower-CPU-overhead, explicitly-managed command submission for high-frequency dispatch | The resident decode / R2 batching fight exactly this overhead |
| `MTLTensor` (`newTensorWithDescriptor:`) | First-class typed multi-dim resource — `Float32`/`Float16`/**`BFloat16`**/`Int8`… | Replaces `DeviceTensor`'s hand-rolled `MTLBuffer` + Python-side shape; native dtype incl. bf16 |
| MSL 4.0 `tensor` type + cooperative (SIMD-group) tensor ops | Hand-written kernels that target the matrix/tensor units directly | The real perf frontier — fused attention/matmul with fusion control MPSGraph can't give |
| MetalPerformancePrimitives cooperative ops — `mpp::tensor_ops::matmul2d` **and `convolution2d`** | Apple's matrix-unit GEMM **and conv**, both with a `cooperative_tensor` destination for in-register bias/activation epilogues | matmul2d → M6/M7/M8 (used); **`convolution2d` is the open rung** — conv2d/conv3d still run on MPSGraph (see integration review P8) |
| `MTL4Archive` / `MTL4BinaryFunction` | Precompiled pipeline archives — faster kernel load | The MSL kernel cache (P4, opt-in) |

## Coexistence strategy

- **Capability gate.** A runtime probe (`tessera_apple_gpu_metal4_available`)
  reports whether MTL4 + MTLTensor + MSL 4.0 are usable; all MTL4 code is behind
  `if (@available(macOS 26.0, *))` so the runtime still compiles + runs on
  older OSes and the non-Darwin stub.
- **MPSGraph stays default.** The MTL4 lane is opt-in per op; anything it doesn't
  cover (or any shape it doesn't handle) falls back to MPSGraph.
- **CI reality.** CI runners are not on Tahoe, so the MTL4 lane is exercised only
  on a Tahoe dev box; tests must skip cleanly when the probe reports unavailable
  (same pattern as the existing `is_metal()` skips).

## The ladder

- **M0 — live capability probe (this PR).** `tessera_apple_gpu_metal4_probe`
  doesn't just read version strings — it actually *creates* the Metal 4 objects
  under `if (@available(macOS 26.0, *))` and reports which succeed:
  `MTL4CommandQueue`, `MTL4CommandAllocator`, `MTL4Compiler`, `MTLTensor`, MSL
  4.0. That is the honest answer to "is the Metal 4 stack usable on this
  machine," and it's the gate every later rung checks. The **full MTL4
  command-model dispatch** (argument tables + residency sets + async-commit
  sync) is substantial and lands with the first real kernel in **M2**, not as a
  throwaway trivial dispatch here.
- **M1 — `MTLTensor`-backed DeviceTensor (this PR).** An `MTLTensor` resource
  variant behind the capability flag, validated against the existing
  buffer-based `DeviceTensor` (round-trip + dtype incl. bf16). The typed-resource
  foundation for MTL4 kernels.
- **M2 — first MTL4 + MSL-4.0 compute kernel (landed).**
  `tessera_apple_gpu_mtl4_scan_f32` runs the Rung-0 scan recurrence as a
  hand-written MSL kernel with a **native in-kernel `for` loop**, dispatched
  through the **full MTL4 command model**: MSL 4.0 library → `MTL4Compiler`
  pipeline → `MTLResidencySet` (explicit residency) → `MTL4ArgumentTable`
  (GPU-address binding) → `MTL4CommandQueue`/allocator/command-buffer/encoder →
  `MTLSharedEvent` CPU sync. Validated bit-close against numpy *and* the MPSGraph
  `forLoop` scan (~1e-7). This is also the concrete **Phase-G → MSL 4.0**
  demonstration (control flow as MSL, not an MPSGraph node).
- **M3 — cooperative-matrix matmul (landed).**
  `tessera_apple_gpu_mtl4_matmul_sg_f32` computes `C = A @ B` with MSL
  `simdgroup_matrix` cooperative tiles — each 32-lane SIMD group computes an 8×8
  output tile via `simdgroup_multiply_accumulate`, hitting the GPU matrix units —
  dispatched through the MTL4 command model (reusing the M2 machinery). M/N/K
  multiples of 8; validated bit-close to numpy (≤1e-7). The first kernel on the
  Metal 4 lane to use the matrix/tensor units directly. (MSL 4.0 also ships a
  newer general `tensor` cooperative-op type — a follow-up; `simdgroup_matrix` is
  the established path.)
- **MTL4 pipeline caching (landed).** The MTL4 compute pipeline (MSL 4.0 compile
  + `MTL4Compiler` pipeline build), the `MTL4Compiler`, and the
  `MTL4CommandQueue` are now created once per `(source, entry)` / per device and
  reused (`compile_mtl4_pipeline` / `mtl4_shared_queue`, mirroring the classic
  `compile_msl_kernel` cache). Cold (first/compile) → warm (cached) for the M3
  matmul: **~58 ms → ~0.46 ms (~125×)**. Because the queue is now shared, each
  call removes its per-call `MTLResidencySet` after sync (the queue's
  residency-set limit is 32). This is the prerequisite for a meaningful
  MTL4-vs-MPSGraph benchmark and for M4 routing.
- **M4 — capability-gated routing (landed, opt-in).** The plumbing to route a
  real op onto the MTL4 lane is in place: `_mtl4_route_matmul_f32` in
  `runtime.py` redirects an eligible `@jit(target="apple_gpu")` matmul to the M3
  `simdgroup_matrix` kernel, gated on (a) the routing flag, (b) the cached MTL4
  capability probe reporting `command_queue` + `compiler`, and (c) the op
  envelope (rank-2 f32, M/N/K multiples of 8). Anything outside the envelope —
  or any non-capable machine — returns `None` and falls through to the MPSGraph
  path unchanged. Toggled by `set_apple_gpu_mtl4_routing(bool)` /
  `TESSERA_APPLE_GPU_MTL4_ROUTE=1`; `apple_gpu_mtl4_routing_enabled()` reads it.
  **Routing is OFF by default, and deliberately so:** a matmul micro-benchmark on
  this M-series Mac (Tahoe) found the MTL4 `simdgroup_matrix` kernel is currently
  **slower than MPS** — MTL4/MPS latency ratios **1.91× (64³), 2.05× (128³),
  2.20× (256³), 3.21× (512³)**. MPS's vendor matmul is hard to beat with a
  straightforward cooperative-tile kernel; flipping the default on would regress
  perf. So M4 ships the *mechanism* (correct, validated bit-close to numpy/MPS
  when enabled) and leaves the *policy* off until a faster MTL4 kernel (e.g.
  MSL 4.0 `tensor` cooperative ops, better tiling/double-buffering) clears MPS.
- **M5 — register-blocked, vectorized GEMM (landed; ~80% of MPS, ties at 1024).**
  The M3 matmul gave one 8×8 output tile to one SIMD group and streamed A/B
  straight from device memory — *zero* data reuse, so every output tile re-read
  its whole row/column band (2–4.8× slower than MPS, widening with size). M5
  rewrites `mtl4_matmul_sg` into **two kernels** chosen at dispatch:
  - **fast path** (`mtl4_matmul_sg_fast`, used when M%64==0, N%64==0, K%16==0):
    a register-blocked GEMM — each threadgroup computes a **64×64** output tile
    with **eight 32-lane SIMD groups** (256 threads, 2×4 layout) each owning a
    32×16 region (a **4×2 array of `simdgroup_float8x8` accumulators**, 8 per
    thread — deliberately ≤8 to avoid register spill). K is walked in **BK=16**
    slabs staged into threadgroup memory with **vectorized `float4` loads** and
    **double-buffered** (next slab prefetched while the current computes).
  - **general path** (`mtl4_matmul_sg`, any other M%8/N%8 shape, any K): a
    bounds-checked 32×32 / 4-SIMD-group double-buffered kernel (zero-pad on load,
    masked store) so correctness holds across all envelope shapes.

  **Measured (M-series, Tahoe, f32, pure GPU-kernel time, best-of-3):** the fast
  kernel hits **~6.6 TFLOP/s** vs the M3/32×32 prototype's ~2.4 — a **~2.8×
  kernel speedup** — reaching **~80% of MPS** (8.2–8.3 TFLOP/s) and matching MPS
  around N=1024. **It still does not beat MPS at 2048+, so routing stays OFF.**
  Knobs swept and rejected: single-buffering, BK∈{8,32}, 128×64/128×128/64×128
  tiles, and threadgroup bank-conflict padding all landed at ~6.0–6.6 TFLOP/s
  (64×64 double-buffered was best); a 64×64 tile with 4×4 (=16) accumulators
  *spills registers* and is ~3× worse. The simdgroup_matrix f32 path has
  **plateaued at ~80% of MPS** here.

  **MSL 4.0 cooperative `tensor` ops (MetalPerformancePrimitives `matmul2d`) —
  investigated, not the f32 answer.** The MPP `mpp::tensor_ops::matmul2d`
  cooperative-tensor op **does compile at runtime** via `newLibraryWithSource:`
  with `MTLLanguageVersion4_0` (the framework headers resolve), but: (1) it has
  **no f32×f32→f32 path under `execution_simdgroups`** — strict/relaxed f32 is
  only supported under `execution_thread` (single-thread, not a fast GEMM). The
  matrix units accelerate **fp16/bf16**, which is where `matmul2d` wins; f32 (our
  production matmul dtype) does not benefit. (2) Even for fp16, the cooperative
  `run` path requires **`tensor_handle` operands** (real `MTLTensor` bound via an
  `MTL4ArgumentTable`), not in-kernel `tensor_inline` views built from a device
  pointer. So adopting MPP means routing **fp16/bf16** matmul onto MTLTensor-bound
  tensor args — a separate integration, and the genuine next rung if the MTL4
  lane is to clear MPS (for half precision). For **f32**, the register-blocked
  simdgroup_matrix kernel above is the realistic ceiling.

  Tested by `tests/unit/test_apple_gpu_metal4.py::test_mtl4_matmul_cooperative_matches_numpy`
  (general + fast paths, partial-tile, multi-threadgroup, and aligned shapes).
- **M6 — fp16 matmul via the MSL 4.0 cooperative `tensor` op (landed; BEATS MPS).**
  M5 established that the `simdgroup_matrix` **f32** path plateaus at ~80% of MPS
  and that MPP `matmul2d` has no f32 cooperative path. M6 takes the rung where the
  tensor units actually pay off: **fp16**. `tessera_apple_gpu_mtl4_matmul2d_f16`
  computes `C[M,N] (f32) = A[M,K] (f16) @ B[K,N] (f16)` with
  `mpp::tensor_ops::matmul2d` — Apple's MetalPerformancePrimitives cooperative
  tensor op — on a **64×64 tile / 4 SIMD groups**, accumulating into a float
  `cooperative_tensor` that is stored to the output. The plumbing that the M5
  investigation found necessary:
  - **Real `MTLTensor` arguments**, not in-kernel `tensor_inline` views (the
    cooperative `run` path rejects pointer-backed tensors). Each operand is a
    **buffer-backed `MTLTensor`** (`[MTLBuffer newTensorWithDescriptor:offset:error:]`)
    with innermost-first extents `(cols, rows)`, packed strides `(1, inner)`,
    `MTLTensorUsageCompute`, bound to an `MTL4ArgumentTable` via
    `setResource:tensor.gpuResourceID atBufferIndex:`.
  - Input tensor element type must be **non-`const`** half (the cooperative-tensor
    type check rejects `const half`), and the destination must be a float
    `cooperative_tensor` (`cT.store` requires matching element type).
  - Compiled at runtime through `compile_mtl4_pipeline` (already sets
    `MTLLanguageVersion4_0`); the MPP framework MSL headers resolve with no extra
    host link. Arbitrary M/N/K — `matmul2d.slice()` edge-checks partial tiles.

  **Measured (M-series, Tahoe, f16→f32, GPU best-of-3, MPP vs MPS, identical
  synced-loop methodology):** MPP **beats MPS** — **3884 vs 3303 GFLOP/s (1.18×)
  at 1024³, 6508 vs 5883 (1.11×) at 2048³**, and **ties at 4096³** (6675 vs 6619).
  A tile sweep confirmed 64×64/4-SIMD-group is the best config (128×64/8 and
  128×128/8 are close; 128×128/4 and 256×128/8 are 3–5× worse — wrong tile:SIMD
  ratio). This is the first MTL4 kernel to **clear MPS**, and the reason is
  precisely that fp16 runs on the matrix units while f32 does not.

  **Routing:** exposed + validated, but **OFF by default** like the rest of the
  MTL4 lane — the GPU-kernel win is ~1.1× and MTL4's per-call dispatch + the f16→
  f32 buffer-copy overhead erode it end-to-end. The mechanism is in place to flip
  fp16 matmul onto this lane if/when the per-call overhead is amortized (resident
  tensors, batched command buffers). Tested by
  `tests/unit/test_apple_gpu_metal4.py::test_mtl4_matmul2d_f16_matches_numpy`
  (aligned, partial, and non-square shapes; f32-accumulated, bit-close to the
  fp16-reference product).
- **M6 (cont.) — bf16 sibling (landed).** `tessera_apple_gpu_mtl4_matmul2d_bf16`
  is the identical MPP `matmul2d` path with `bfloat` input tensors
  (`MTLTensorDataTypeBFloat16`) → f32 cooperative accumulator → f32 output. This
  matters more than fp16: **MPS has no native bf16 GEMM**, so Tessera's existing
  `tessera_apple_gpu_mps_matmul_bf16` falls back to an on-host fp32 conversion.
  The native MPP bf16 tensor op is **~10× that conversion fallback** end-to-end
  (2048³: ~12 ms vs ~149 ms) — bf16 is the clearest case for routing onto the
  MTL4 lane. Correct to ≤6e-2 vs the bf16-reference product (fp32 accumulation).
  Tested by `test_mtl4_matmul2d_bf16_matches_numpy`.
- **M7 — fused epilogue (bias + activation), landed; essentially free.** Because
  the matmul result lands in a float `cooperative_tensor` that is live in
  registers across the SIMD groups, M7 applies **bias (per output column) +
  activation in-register before the single store** — no extra device round-trip.
  `tessera_apple_gpu_mtl4_matmul2d_epilogue_{f16,bf16}(A, B, C, bias, act, M,N,K)`
  with `act` ∈ {none, relu, gelu(tanh), silu}; Python
  `apple_gpu_mtl4_matmul2d_epilogue(..., bias=, act=, dtype=)`. The per-element
  walk uses the real cooperative_tensor API — `get_capacity()`,
  `is_valid_element(i)` (the actual mask accessor; `get_mask` does not exist),
  `operator[]`, and `get_multidimensional_index(i)` whose `index[0]` is the
  N/column axis (locked by `test_mtl4_matmul2d_epilogue_bias_is_per_output_column`,
  which sets A=B=0 and asserts the result equals the per-column bias). **Measured
  cost of the fused epilogue vs the bare matmul: ~0% (+6% at 1024³, −10% at 2048³
  — noise).** That is the whole point: the alternative is a second elementwise
  kernel that re-reads and re-writes the full M×N output (a memory-bound round
  trip the fusion eliminates). Correct for both dtypes × all four activations
  (tested by `test_mtl4_matmul2d_epilogue_fuses_bias_and_activation`). Routing
  stays OFF by default (same per-call-overhead reasoning as M6); the fused kernel
  is the natural thing to route a `linear → bias → activation` block onto once the
  lane is amortized, since it collapses three ops into one dispatch.
- **M8 — resident-weight MLP-block session (landed; amortizes the overhead that
  kept routing off).** A `linear → bias → activation` block is *already* a single
  `matmul2d` epilogue dispatch (M7) — `Y = act(X @ W + b)`. M8 makes that practical
  for **decode** (repeated small-M steps with persistent weights) by keeping `W`,
  bias, params, pipeline, residency set, and command queue **resident/reused
  across calls**; each `run(X)` only uploads the small activation and dispatches.
  C ABI `tessera_apple_gpu_mtl4_mlp_session_{create,run,destroy}` (opaque handle =
  ARC-managed C++ struct); Python `runtime.AppleGPUMLPSession(W, np, bias=, act=,
  dtype=)` with `.run(X)`, context-manager + `close()`, and a numpy fallback when
  Metal 4 is unavailable. **Measured (M-series, Tahoe, f16, K=N=4096):** per-step
  latency **session vs per-call epilogue = 3.3–3.6× faster** (M=1/8/64) — precisely
  because the per-call path re-uploads the 32 MB weight and re-commits residency
  every step while the session does neither. This is the lever that lets the lane
  beat MPS at decode sizes (where per-call MTL4 overhead, not kernel throughput,
  was the gap). Tested by `test_mtl4_mlp_session_resident_weights_matches_reference`
  / `_matches_oneshot_epilogue` / `_run_after_close_uses_fallback`.

  **nn.functional status.** The fused `linear+bias+activation` *kernel* and the
  resident-weight *session* are the building blocks; the remaining follow-up is
  **compile-time Graph IR chain recognition** so a user writing
  `gelu(linear(x, W, b))` under `@jit(target="apple_gpu")` in f16/bf16 auto-lowers
  to one `matmul2d_epilogue` dispatch (today it composes through the per-op MPSGraph
  path unless the epilogue API / session is called explicitly). That recognizer
  mirrors the existing `matmul→gelu` / `matmul→rmsnorm` chain detection in
  `runtime.py` + `driver.py`, extended with a bias operand and the f16/bf16 gate.

## P-series — overhead amortization + first default routing flip (landed)

Following the integration review below: **P1** put the MTL4 lane on the shared
buffer pool; **P2/P3** made the command allocator / command buffer / argument
table / shared event reusable (reset+rebound per dispatch, serialized by
`mtl4_dispatch_mu`) — repeated small epilogue dropped 0.61 ms → 0.28 ms; **P5**
routes `@jit(target="apple_gpu")` **bf16 matmul to the native `matmul2d` tensor-op
by default** (`_mtl4_route_matmul2d_bf16`, toggle `TESSERA_APPLE_GPU_MTL4_BF16`),
measured **11.8–14.7× faster** than the legacy fp32-conversion fallback — the
first dtype routed onto the MTL4 lane by default. (f32 stays opt-in; MPS's f32
GEMM is well-tuned.) **P6**: `gelu(linear(x,W,b))` under `@jit(target="apple_gpu")`
now auto-fuses — the driver recognizes `matmul→add(bias)→{gelu,relu,silu}` and the
runtime dispatches f16/bf16 to one `matmul2d` epilogue (f32 / residual-add keep
the matmul on MPS). **P4**: opt-in `MTL4Archive` pipeline persistence
(`apple_gpu_mtl4_archive_enable(path)` / `_flush()`) so pipelines survive process
restarts. The integration review is essentially closed; only minor cleanup
remains (pool the M2 scan + M3/M5 matmul, reuse the residency-set object).

## Integration review

A Metal-4-grounded audit of how the whole Apple GPU backend uses the device
(singletons, buffer pool, graph/pipeline caches, the per-call MTL4 overhead, and
the unused `MTLArchive`/ML-encoder surfaces) lives in
[apple_backend_integration_review.md](apple_backend_integration_review.md). Headline:
the kernels are competitive; the remaining gap to "most optimal" is per-call
host-side overhead amortization (P1 buffer-pool — partially done; P2/P3 reuse the
command allocator/buffer/argument-table/shared-event), which the M8 session
already validated as worth ~3.3× at decode sizes.

## Trade-offs + risks

- **Hand-written kernels.** MPSGraph handles arbitrary shapes/dtypes and fuses
  for us; the MTL4 lane is hand-written MSL — more control, more maintenance, and
  per-shape coverage is on us.
- **OS-gated.** Everything requires macOS 26+. The runtime must degrade cleanly
  everywhere else (probe → unavailable → MPSGraph).
- **Measured perf delta (so far: MPS wins).** The M4 matmul micro-benchmark found
  the MTL4 `simdgroup_matrix` kernel 1.9–3.2× *slower* than MPS on this Mac, so
  routing stays opt-in/off. The dispatch-overhead and tensor-core wins are real in
  principle but a kernel must actually clear MPS before the default flips.
- **Not a migration.** This never removes the MPSGraph lane. If the MTL4 lane
  doesn't pay off, it stays a capability-gated experiment with zero cost to the
  default path.

## Acceptance (M0 + M1)

- `tessera_apple_gpu_metal4_probe` returns honest capability bits and runs a
  trivial compute through the MTL4 command model, numerically validated; returns
  "unavailable" cleanly off Tahoe / non-Darwin (stub).
- An `MTLTensor`-backed `DeviceTensor` round-trips f32/f16/bf16 against the
  buffer-based path behind the capability flag.
- Unit tests skip when the probe reports unavailable; no regression to the
  MPSGraph lane.

## Acceptance (M4)

- Routing is OFF by default; `apple_gpu_mtl4_routing_enabled()` reflects the flag
  and `set_apple_gpu_mtl4_routing(bool)` / `TESSERA_APPLE_GPU_MTL4_ROUTE` toggle it.
- With routing enabled, an eligible (rank-2 f32, 8-multiple M/N/K)
  `@jit(target="apple_gpu")` matmul routes onto the MTL4 lane and returns a result
  bit-close to numpy/MPS; out-of-envelope shapes/dtypes and non-capable machines
  fall through to MPSGraph unchanged.
- `_mtl4_route_matmul_f32` returns `None` (not a wrong result) whenever the flag is
  off, the dtype/shape is ineligible, or the capability probe is missing.
