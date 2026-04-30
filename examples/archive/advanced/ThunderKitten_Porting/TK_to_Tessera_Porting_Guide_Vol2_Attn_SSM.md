<!-- MERGE_START: TK_to_Tessera_Porting_Guide -->
# ThunderKittens → Tessera Porting Guide (Vol. 2: Attention & SSMs)

**Date:** 2025-09-02  
**Scope:** Flash‑style attention (fwd/bwd), linear attention variants, Mamba/selective‑scan, and long convolution (FFT) mappings.

---

## 1) Flash‑style Attention (FA‑2/FA‑3 flavor)

**TK structure:** tiled `Q@Kᵀ` → rowwise softmax → `P@V`, with double‑buffered smem and overlapping TMA & MMA.  
**Tessera mapping:**

1. Tile‑load `Q,K,V` with `async_copy_2d`; overlap via `pipeline.commit()/wait()` across two smem buffers.  
2. Compute `S = Q@Kᵀ` using `mma_sync` with FP32 accum.  
3. Rowwise `rmax`, subtract, `exp2`, rowwise sum, reciprocal, and multiply (stream‑softmax).  
4. Compute `O = P@V` via `mma_sync`; optional epilogue chain (bias, dropout, residual).

**Skeleton:**

```cpp
// Pseudo‑Tessera FlashAttention forward
tile<float, 16, 16> s = tile_zero<tile<float,16,16>>();
s = mma_sync(q, kt, s);                    // S = Q @ K^T

auto m = tile_reduce_rowmax(s);
s = s - broadcast_row(m);                  // subtract rmax
s = exp2(s);
auto l = tile_reduce_rowsum(s);
auto p = s * rsqrt(l + eps);               // normalize

tile<float,16,16> o = mma_sync(p, v);
o = epilogue_chain(o, maybe_dropout(pdrop, seed), maybe_bias(), maybe_residual(x));
```

### Backward pass

- Reuse forward intermediates or recompute “cheap” terms.  
- Use tilewise reductions for dQ/dK/dV; maintain numerically stable softmax backward (avoid catastrophic cancellation).  
- Fuse dropout mask reuse if deterministic mode is required.

---

## 2) Linear attention variants

**Kernel pattern:** per‑row feature map `φ(.)` then `Q' = φ(Q)`, `K' = φ(K)`, `V' = V`, accumulate `K'ᵀ V` (global), and compute `O = Q' (K'ᵀ V)` with normalization.  
**Tessera helpers:** `tile_map_rows(φ)`, `tile_reduce(colsum)`; a convenience `tile_scan` is nice for some kernels but optional.

- **Checklist:** watch numeric growth of feature maps; use FP32 accum; optionally clip or rescale per block.

---

## 3) Mamba‑2 / selective scan

**Pattern:** gated selective‑scan across sequence dimension with short filters per channel.

- Express as a fused loop: load tile, apply gates, perform per‑row scan/reduction; keep state in registers.  
- Desirable helper: `selective_scan(tile x, tile g, /*params*/)` producing the fused recurrence.  
- Validate against reference PyTorch/NumPy scans; check warm‑up state handling at block boundaries.

---

## 4) Long convolution via FFT (FlashFFTConv‑style)

**Plan:** expose a thin Tessera runtime wrapper over cuFFT (R2C/C2R) + a tiled complex pointwise multiply kernel.

- **Forward:** pad→FFT(x), FFT(k), pointwise multiply (complex), IFFT→crop.  
- **Backward:** reuse cached plans; conjugate multiply; manage scaling.  
- **Nice‑to‑have:** `ops.signal.fft_conv1d(x, k, mode="same")` with autotuned tile sizes for the pointwise multiply.

---

## 5) Validation & performance

- **Correctness:** compare across multiple seq_lens and head_dims; verify dropout mask determinism when seeded.  
- **Perf:** ensure overlap of TMA with MMAs in both QKᵀ and PV phases; watch shared‑memory pressure against occupancy.

---

## 6) Ready‑to‑port list (Vol. 2)

- FlashAttention fwd/bwd, linear attention variants, Mamba/selective scan, long convolution (FFT wrapper + pointwise kernel).

<!-- MERGE_END: TK_to_Tessera_Porting_Guide -->
