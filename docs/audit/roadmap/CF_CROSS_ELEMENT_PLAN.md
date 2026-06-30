---
last_updated: 2026-06-30
audit_role: plan
plan_state: open
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

### CF4d-2 — norm-in-loop *(done, ROCm/gfx1151)*
Body `rmsnorm(carry)` / `layer_norm(carry)` over a 1xK carry →
`GenerateROCMControlForNormKernel` (`--generate-rocm-control-for-norm-kernel`),
the same cooperative substrate as CF4d-1: carry in LDS, `gpu.barrier` per
iteration. Because K ≤ BD (one element per thread), each thread reads the K LDS
values and computes the statistic in-register, then normalizes its own element —
no inter-thread reduction op, just the barrier handoff. `rmsnorm`:
`x / √(mean(x²) + eps)`; `layer_norm`: `(x − μ) / √(mean((x−μ)²) + eps)` (two
in-register reductions). `eps` baked from the op attr. Proven on gfx1151 by
`tests/unit/test_rocm_control_for_norm_exec.py` (looped rmsnorm/layer_norm vs the
same numpy formula applied iteratively).

### CF4d-3 — WMMA single-tile matmul *(done, ROCm/gfx1151)*
Body `matmul(carry, W)` where carry and W are **16×16 f16 tiles** (one RDNA3.5
WMMA `16×16×16`) → `GenerateROCMControlForWmmaKernel`
(`--generate-rocm-control-for-wmma-kernel`), one wave. Emits the
`rocdl.wmma.f32.16x16x16.f16` intrinsic directly. The new piece vs CF4d-1/2 is
the **accumulator→input fragment shuffle through LDS** between iterations: the
WMMA result is a `vector<8xf32>` accumulator fragment, written back to LDS in
LOGICAL `[row][col]` order (`acc c[e] → lds[(2e+lhi)*16 + lane]`) and re-read as
the `vector<16xf16>` A-fragment (`a[i] = lds[lane*16 + i]`); the B-fragment
(`b[i] = W[i*16 + lane]`) is loop-invariant. Because both the store and load
index the matrix logically, the LDS is a plain 16×16 f16 matrix and the handoff
is layout-correct. Proven on gfx1151 by
`tests/unit/test_rocm_control_for_wmma_exec.py` (`carry @ W^it`, it=1/2/3,
mirroring the per-iteration f16 truncation; f16 tolerance). Larger carries are
multi-tile — CF4d-4 (one workgroup) / CF4d-5 (multi-workgroup).

### CF4d-4 — multi-tile, one workgroup *(done, ROCm/gfx1151)*
Carry is an **M×K tile grid** (M, K multiples of 16) and W is **K×K**, both f16 →
`GenerateROCMControlForWmmaTileKernel`
(`--generate-rocm-control-for-wmma-tile-kernel`). The CF4d-3 insight that makes
this cheap *without* a grid-wide barrier: a multi-tile carry still fits in **one
workgroup's LDS** (e.g. 64×64 f16 = 8 KB ≪ 64 KB), so we launch **one workgroup
of MT·KT waves** — each wave owns one 16×16 output tile `(ti, tj)` and
accumulates it over the shared-K dimension,
`D[ti][tj] = Σ_tk carry[ti][tk] @ W[tk][tj]`, chaining KT WMMA ops into one
accumulator fragment. The whole carry lives in LDS as a plain M×K f16 matrix; per
control-loop iteration every wave reads its row-block, a **workgroup barrier**
fences all old-carry reads, each wave writes its output tile back to LDS, a
second barrier publishes the new carry. Two barriers, **no cross-workgroup
sync** — the B-fragments are loop-invariant (W fixed) so they're built once up
front. Proven on gfx1151 by `tests/unit/test_rocm_control_for_wmma_tile_exec.py`
(32×32 it=1/2/3, plus asymmetric 16×32 / 32×16 and the 8-wave-ceiling 64×32 /
32×64).

**On-device envelope (measured):** one workgroup must be co-resident on a single
WGP, so the tile grid is bounded by how many of *this* kernel's waves fit there —
not the 1024-thread/workgroup hardware max. The WMMA accumulator + A/B fragments
push VGPR/wave high enough that gfx1151 holds **8 waves (256 threads)** per WGP;
9+ waves fail to launch (`hipErrorLaunchFailure`, confirmed on-device). The pass
therefore **caps `MAX_WAVES = 8`** and leaves larger carries untouched — it never
emits a kernel that can't launch on the target (Decision #21). That covers up to
a 4×2 / 2×4 / 2×2 tile carry (e.g. 64×32, 32×64, 32×32).

### CF4d-5 — multi-workgroup (grid.sync frontier)
Carry exceeding one WGP's wave capacity (>8 tiles here) → iteration N+1 needs
**all** of N's tiles across **multiple workgroups**, so a **grid-wide barrier** is
required (`grid.sync` + `hipLaunchCooperativeKernel`). This is the only step that
needs true cross-workgroup sync; deferred until a real workload demands carries
that large.

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
