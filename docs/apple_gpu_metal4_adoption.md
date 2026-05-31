# Metal 4 adoption — design + ladder

> Status: **M0 + M1 + M2 + M3 + M4 + M5 + M6 landed.** This machine runs macOS 26.5 (Tahoe) with
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
| `MTL4Archive` / `MTL4BinaryFunction` | Precompiled pipeline archives — faster kernel load | The MSL kernel cache |

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
