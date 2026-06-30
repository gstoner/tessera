---
status: Active plan
classification: Roadmap
authority: CONTROL_FLOW_AND_DEEPSEEK_ACCELERATION_PLAN.md (CF4d extension)
last_updated: 2026-06-30
---

# Cross-Element Control-Flow Device Kernels (CF4d) — Plan

> **Status:** active. ROCm-led on gfx1151 (RDNA3.5). Builds on the completed
> elementwise control-flow track (CF0–CF4c, #224–#236). Spec cross-link:
> `docs/spec/CONTROL_FLOW_CONTRACT.md`.

## Why this exists

The shipped device control-flow kernels (`GenerateROCMControlForKernel`) use a
**per-thread** model: one thread owns one carry *element* and runs the whole
recurrence locally (`scf.for`/`scf.if`/`scf.while` over a scalar). That model is
correct and bit-exact for **elementwise** bodies (`add`, `silu`, …) — but it
**cannot express a body that mixes elements**: `matmul`, `softmax`, `rmsnorm`,
`layer_norm` all need *every* carry element to produce *each* output element (a
reduction across the carry). A single thread doesn't have the other elements.

These are exactly the bodies that matter for real models:

- **GEMV / matmul recurrence** — `h = h @ W` per step → RNN hidden state, linear
  attention / RetNet state update, SSM/Mamba scan.
- **norm-in-loop** — `h = rmsnorm(h)` / `layer_norm(h)` per step → normalized
  recurrences, gated state.

So CF4d adds a **cooperative-workgroup** kernel family: one workgroup owns the
whole carry vector in **LDS** (shared memory), threads cooperate on the
cross-element op, and a `gpu.barrier` separates iterations. The whole bounded
loop is still **one device dispatch**.

## The substrate (shared by every CF4d step)

The reduction kernels already establish the pattern we reuse
(`GenerateROCMReduceKernel` / `Norm` / `Scan`):

- **LDS carry buffer:** `gpu::AddressSpaceAttr::get(ctx, Workgroup)` +
  `gpuFunc.addWorkgroupAttribution(memref<KxT, #workgroup>, loc)`.
- **Workgroup barrier:** `gpu::BarrierOp::create(b, loc)`.
- **One workgroup per carry**, `blockDim = BD` threads, grid `= 1` (single
  carry) or `= num_carries` (batched, one workgroup each — still no *global*
  sync needed because each carry is independent).

The bounded loop is an outer `scf.for %i = 0 to max_iters`; each iteration reads
the LDS carry, computes the cross-element op cooperatively, barrier, writes the
new carry to LDS, barrier. Lowers through the proven chain
`convert-scf-to-cf → convert-gpu-to-rocdl → rocdl-attach-target{gfx1151} →
gpu-module-to-binary → hsaco → hipModuleLaunchKernel`.

## Steps (incremental, one PR each)

### CF4d-1 — GEMV recurrence *(this PR)*
Body is a single `matmul(carry, W)` where **carry is a vector** `1×K` and `W` is
`K×K`, so `carry @ W` is a GEMV (`1×K @ K×K = 1×K`) — cross-element (each output
`o[j] = Σ_k carry[k]·W[k][j]`) but **no WMMA fragments**, just LDS + a reduction
loop. Kernel ABI `(CARRY, W, OUT : memref<?xf32>, K : index)`:

```
gpu.func @ctrl_for_gemv(%CARRY, %W, %OUT: memref<?xf32>, %K: index) kernel {
  %lds = workgroup memref<BDxf32>          // carry in LDS
  // load carry → lds[tid] for tid < K ; barrier
  scf.for %i = 0 to max_iters {
    // thread j (j<K): acc = Σ_k lds[k] * W[k*K + j]   (inner scf.for over K)
    // barrier ; lds[j] = acc ; barrier
  }
  // OUT[j] = lds[j]
}
```
Numeric proof on gfx1151: `carry @ W^max_iters` vs numpy. Single-tile (K ≤ BD),
one workgroup, no global sync. This establishes the cooperative substrate.

### CF4d-2 — norm-in-loop
Body `rmsnorm(carry)` / `layer_norm(carry)`: a workgroup reduction
(`Σ carry²` / mean+var) in LDS, then a normalize pass, looped. Reuses the
`GenerateROCMNormKernel` reduction structure inside the control loop.

### CF4d-3 — WMMA single-tile matmul
Body `matmul(carry, W)` where carry is a **16×16 tile** (one RDNA3.5 WMMA
`16×16×16`). Reuses the `GenerateWMMAGemmKernel` fragment machinery; the only
new piece is the **accumulator→input fragment shuffle through LDS** between
iterations (D-fragment f32 → write LDS → read back as the A-fragment f16). One
wave, no global sync.

### CF4d-4 — multi-tile (cooperative kernel)
Carry larger than one tile/workgroup → iteration N+1 needs **all** of N's tiles,
so a **grid-wide barrier** is required (`grid.sync` / cooperative-launch). This
is the only step that needs cross-workgroup sync; deferred until CF4d-1..3 land.

## Validation discipline (every step)
- A hardware-free **lit** fixture (`// REQUIRES: tessera-rocm-backend`) checking
  the emitted `gpu.func` structure (LDS attribution, barriers, the loop).
- An on-gfx1151 **execute** test (`tests/unit/test_rocm_control_*_exec.py`,
  skip-clean off-GPU) comparing to a numpy reference with a closed form
  (`carry @ W^max`, etc.).
- Validation `validateGemvBody` (etc.) only emits a kernel for the exact
  supported shape/op; anything else is left for the CF0 guard / SCF — never a
  silently-wrong kernel.

## How this supports kernel development
The cooperative-workgroup + LDS + barrier scaffold CF4d-1 lands is the **reusable
substrate** for every cross-element control body. CF4d-2/3 specialize the
in-loop op (reduction / WMMA) on top of it; the control-loop wrapper, the LDS
carry handoff, the barrier discipline, and the execute-test harness are written
once here. Downstream kernel work (linear-attention state, SSM scan, gated
recurrences) targets this substrate instead of re-deriving the loop + LDS plumbing.
