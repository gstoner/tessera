---
last_updated: 2026-08-04
audit_role: reference
owning_plan_item: ROCm GEMM codegen
---

# ROCm GEMM performance ladder — measured, with the implementation spec

Closing the gap between the compiler-generated GEMM and the hand-written
kernels. Every number here is measured on gfx1151 (32 CU @ 2.9 GHz, LPDDR5X-8000
/ 256-bit ≈ 256 GB/s) at **2048³ f16, kernel-only**, through
`benchmark_rocm_wmma_gemm.py` after the timing fix in PR #512.

**All measurements are `timer_source = host_wall`** — the HIP event API on this
WSL2 / `/dev/dxg` host returns garbage while reporting success, so the harness
falls back to a synchronized host clock. Wall includes launch overhead, so these
figures are *conservative*: true kernel TFLOP/s is ≥ what is shown. Rankings are
unaffected because every row shares one modality.

---

## 1. The ladder

| increment | block tile | AI (FLOP/B) | TFLOP/s | Δ |
|---|---|---:|---:|---|
| naive register schedule (production) MT=2,NT=4 | — | — | **8.02** | — |
| LDS 1 wave, 1×1 tiles (what the generator emits) | 16×16 | 8.0 | 2.95 | — |
| + register tiling 2×2 | 32×32 | 16.0 | **5.01** | **+70%** |
| + register tiling 4×4 | 64×64 | 32.0 | 6.07 | +21% |
| + wave tiling 2×2 (2×2 tiles) | 64×64 | 32.0 | **8.76** | +44% |
| + wave tiling 4×2 (2×2 tiles) | 128×64 | 42.7 | 9.90 | +13% |
| + double buffering (`bench_pipe`) | 128×64 | 42.7 | **10.29** | +4% |

Arithmetic intensity for a BM×BN block with depth BK in fp16:

$$\mathrm{AI}=\frac{2\,BM\cdot BN\cdot BK}{2(BM\cdot BK+BK\cdot BN)}=\frac{BM\cdot BN}{BM+BN}$$

## 2. Two results that change the build order

**Register tiling is the cheapest large win.** 1×1 → 2×2 is +70% and is the
smallest code change in the ladder: more accumulators and more WMMA calls, no
wave partitioning, no double buffering.

**At equal AI, wave tiling beats deeper register tiling — 8.76 vs 6.07, both at
AI 32.** AI predicts *bandwidth*; it says nothing about register pressure. A 4×4
register block needs 16 accumulators × 8 VGPRs = 128 accumulator VGPRs per lane,
which collapses occupancy. Spreading the same block across waves keeps the
per-wave register footprint small and the CU busy. **Do not pick a tiling by AI
alone.**

**Double buffering is +4%**, not a headline. It is the last rung, not the
second.

So the build order is **register tiling → wave tiling → double buffering**,
which is not the order the work was originally framed in.

## 3. Implementation spec — `emitCanonicalLdsBody`

Current state: one-wave, MT=NT=1, and the caller hard-errors on `mt != 1 ||
nt != 1`. It carries **one** accumulator through the K-loop and issues **one**
`tessera_rocm.wmma`.

### Step 1 — register tiling (MT×NT)

| what | now | target |
|---|---|---|
| `ldsA` | `memref<256 x T>` | `MT*16*16` |
| `ldsB` | `memref<256 x T>` | `16*NT*16` |
| `baseRow` | `blockIdx.y * 16` | `blockIdx.y * (MT*16)` |
| `baseCol` | `blockIdx.x * 16` | `blockIdx.x * (NT*16)` |
| copyA bound | `256` | `MT*256`; `row = e/16`, `kk = e%16` |
| copyB bound | `256` | `NT*256`; `k = e/(NT*16)`, `col = e%(NT*16)` |
| `scf.for` iter_args | 1 accumulator | `MT*NT` accumulators |
| A fragments | 1 | `MT`, contiguous 16 at `(mi*16 + lane)*16` |
| B fragments | 1 | `NT`, element `i` at `i*(NT*16) + ni*16 + lane` |
| WMMA calls | 1 | `MT*NT`, reusing each A and B fragment |
| epilogue | 8 elements | `MT*NT` × 8, at row `baseRow + mi*16 + 2e + lhi`, col `baseCol + ni*16 + lane` |

Lift the `mt != 1 || nt != 1` guard in the caller once this lands.

### Step 2 — wave tiling (WM×WN)

Block tile becomes `WM*MT*16 × WN*NT*16`. Each wave owns one `MT*NT` sub-block;
`waveId = tx / 32`, `waveRow = waveId / WN`, `waveCol = waveId % WN`. The LDS
panels grow accordingly and the cooperative copy loops stride over
`nthreads = WM*WN*32`.

### Step 3 — double buffering (optional, +4%)

Two LDS buffers, prefetch panel `p+1` during the compute of `p`, one barrier per
panel. See `kKernelTemplatePipe`.

## 4. What the oracle tells us we do *not* need

`kKernelTemplatePipe` reaches 10.29 TFLOP/s using **scalar** LDS reads and
**scalar** grid-strided global→LDS copies — no `dwordx4`, no XOR swizzle. The
entire 3.5× is data-reuse structure, not instruction-level tuning. Vectorized
loads and bank-conflict avoidance are a later, separate lever.

## 5. Verification gate

Not bit-identity — a different reduction order legitimately changes results.

1. **Numerics**: error vs an fp64 reference within the inner-product bound
   $|fl(x^Ty)-x^Ty|\le\gamma_K|x|^T|y|$, $\gamma_K=Ku/(1-Ku)$. With fp32
   accumulation ($u=2^{-24}$) and K=2048, $\gamma_K\approx1.2\times10^{-4}$.
2. **Differential**: against `tessera_rocm_wmma_gemm_f16_bench_{lds,pipe}`,
   which are callable oracles.
3. **Throughput**: against this ladder, at the same `timer_source`.

A structural fixture that only checks emitted ops is *not* sufficient — it
passes while the kernel returns a partial product.
