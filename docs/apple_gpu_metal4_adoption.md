# Metal 4 adoption — design + ladder

> Status: **M0 + M1 + M2 + M3 landed.** This machine runs macOS 26.5 (Tahoe) with
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
