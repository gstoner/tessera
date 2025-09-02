<!-- MERGE_START: TK_to_Tessera_Porting_Guide -->
# ThunderKittens → Tessera Porting Guide (Vol. 1: Core Kernels)

**Date:** 2025-09-02  
**Scope:** Practical mapping of TK’s tile-centric kernels (GEMM, epilogues, RoPE, (RMS)LayerNorm, basic reductions) to the Tessera Programming Model.

---

## 1) Quick mental model

TK and Tessera both operate on **register tiles** backed by **shared-memory staging** with **async global→smem copies** and **tensor‑core MMAs**. The mapping is mostly mechanical:

- **TK 16×16 register tiles** → **Tessera tile objects** (`tile<f16,16,16>`, `tile<bf16,16,16>`, etc.).  
- **TK TMA async copies** → **Tessera `async_copy_2d/3d`** with pipeline stages and barriers.  
- **TK WGMMA/WMMA** → **Tessera `mma_sync` tile intrinsics** lowered to target (Hopper/Blackwell).  
- **TK LSCF pipeline (Load→Compute→Store→Finish)** → **Tessera pipeline API** (`pipeline.begin/commit/wait`) around ping‑pong smem buffers.  

> Result: algorithmic structure remains intact; only intrinsics and helpers change.

---

## 2) Mapping cheat sheet (by kernel family)

| TK pattern | Tessera equivalent | Notes |
|---|---|---|
| GEMM (bf16/fp16/fp8) | `ops.linalg.gemm(tile…)` + tunable `mma_kind`, `cta_pairs`, `split_k` | FP8 path uses scale/dequant epilogue hooks. |
| Bias / GELU / SiLU epilogues | `ops.epilogue.chain([bias, act])` | Chainable tile‑epilogues keep data in registers. |
| (RMS)LayerNorm | `ops.norm.rms/ln` with `tile_reduce(mean,var)` | Fuse with residual/dropout via epilogue chain. |
| RoPE | `ops.attn.rope(q_or_k)` | Implement as in‑place 2×2 rotations per head‑dim stride. |
| Reductions (row/col) | `tile_reduce(row|col, op)` | Backed by warp shuffles + smem as needed. |

---

## 3) Porting checklist (core kernels)

1. **Tile shapes & dtypes**: choose `M,N,K` multiples that match tensor‑core fragment shapes for the target (e.g., 16×16 for bf16/fp16).  
2. **Memory layout**: map TK’s smem layout to Tessera’s `smem_layout_t` (row‑major vs swizzled); ensure coalesced global reads.  
3. **Async copies**: convert `tma.async_copy` to `async_copy_2d/3d` with `pipeline.commit()` and appropriate `barrier.arrive()/wait()`.  
4. **MMA loop**: replace TK `wgmma.mma_async` with `mma_sync(tileC, tileA, tileB, accum=tileC)`; schedule split‑K if needed.  
5. **Epilogues**: migrate to `ops.epilogue.chain`; prefer fusing bias/activation/residual/dropout to avoid extra writes.  
6. **Numerics**: match TK accumulation types (e.g., bf16→fp32 accum). Validate with bit‑for‑bit or tolerance tests.  
7. **Launch**: mirror CTA tiling and grid mapping; enable `cta_pairs` when on Blackwell (B100/B200).  
8. **Perf sanity**: roofline‑check (HBM BW vs TC flops), occupancy, smem bank conflicts, and TMA issue rate.

---

## 4) GEMM skeleton (Tessera‑style, illustrative)

```cpp
// Pseudo‑Tessera kernel sketch (C++/CUDA‑like API)
__global__ void gemm_bf16_tile(const bf16* A, const bf16* B, float* C,
                               int M, int N, int K, float alpha, float beta) {
  using tileA_t = tile<bf16, 16, 16>;
  using tileB_t = tile<bf16, 16, 16>;
  using tileC_t = tile<float,16,16>;

  __shared__ smem_t<tileA_t, 2> sA;  // ping‑pong
  __shared__ smem_t<tileB_t, 2> sB;

  pipeline_t pipe;
  tileC_t c = tile_zero<tileC_t>();

  for (int k0 = 0; k0 < K; k0 += 16) {              // K‑slice loop
    int buf = (k0 / 16) & 1;

    async_copy_2d(sA[buf], &A[/*global offset*/], /*ldA*/);
    async_copy_2d(sB[buf], &B[/*global offset*/], /*ldB*/);
    pipe.commit();               // make copies visible
    pipe.wait(buf);              // ensure sA/sB[buf] ready

    tileA_t a = smem_load<tileA_t>(sA[buf]);
    tileB_t b = smem_load<tileB_t>(sB[buf]);
    c = mma_sync(c, a, b);       // tensor‑core MMA
  }

  // Epilogue: beta*C + alpha*c, then optional bias/activation
  tileC_t out = epilogue_chain(c, make_scale(alpha), maybe_bias(), maybe_gelu());
  global_store(out, &C[/*global offset*/], /*ldC*/);
}
```

> **Tip:** Maintain TK’s split‑K and CTA‑pairing where applicable—Tessera exposes equivalent launch knobs.

---

## 5) RoPE & (RMS)LayerNorm notes

- **RoPE:** Use a compact in‑register 2×2 rotation for even/odd channels, fused into Q/K pre‑processing.  
- **RMSNorm/LayerNorm:** Perform rowwise `mean`/`var` via `tile_reduce(row, …)` and fuse scale/shift; prefer FP32 accumulators.

---

## 6) Validation strategy

- **Correctness:** start with tiny, odd shapes; compare against cuBLAS/cuDNN or a high‑precision reference.  
- **Numerics:** track max ULP error and relative error; confirm determinism when required (Philox seed for dropout).  
- **Performance:** add NVTX ranges; record achieved TFLOP/s and HBM GB/s; ensure overlap of TMA and MMA phases.

---

## 7) Ready‑to‑port list (Vol. 1)

- GEMM (bf16/fp16/fp8), Bias+Activation epilogues, RoPE, (RMS)LayerNorm, simple reductions.  
- Symmetric matmul (triangular mask) — add a `triangular_only` schedule flag.

<!-- MERGE_END: TK_to_Tessera_Porting_Guide -->
