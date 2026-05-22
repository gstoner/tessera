<!-- MERGE_START: Tessera_Enhancement_Proposals -->
# Tessera Enhancements (Programming Model, Compiler, Runtime) — Sept 2025

**Goal:** Smooth TK→Tessera ports, unlock Blackwell‑class performance, and simplify authoring of attention/SSM kernels.

---

## A) Programming Model (language & intrinsics)

1. **Epilogue chaining as first‑class**  
   - `ops.epilogue.chain([bias, act, dropout, residual])` with compile‑time fusion guarantees.  
   - Typed FP8 scale/dequant nodes (`e4m3`, `e5m2`) with per‑tile scale vectors.

2. **Convenience tile collectives**  
   - `tile_reduce_rowmax`, `tile_reduce_rowsum`, `tile_softmax_rmax(eps)` (stable path).  
   - `tile_scan` + `selective_scan` helpers for Mamba‑style recurrences.  
   - `triangular_only(mask=lower|upper)` schedule flag for symmetric/triangular GEMMs.

3. **Pipelines made easy**  
   - `pipeline(slot_count=2|3)` RAII wrapper; `slot.next()` swaps buffers and synchronizes appropriately.

4. **Numerics & determinism**  
   - Dropout with Philox seeding wired through kernels; `deterministic=True` guarantees mask reuse on bwd.

---

## B) Compiler (IR & passes)

1. **Tile‑IR extensions**  
   - WGMMA variants (bf16/fp8) with CTA‑pair codegen.  
   - TMA 2D/3D ops + descriptor hoisting/caching.  
   - FFT placeholder ops lowered to cuFFT runtime calls when available.

2. **Fusion & scheduling passes**  
   - Epilogue fusion (bias/act/dropout/residual) into GEMM/attention pipelines.  
   - Attention‑pattern recognizer (QKᵀ‑softmax‑PV) for auto‑pipelining.  
   - Split‑K and CTA‑pair auto‑selection via cost model or autotuner.

3. **Attributes & constants**  
   - Small pass to **resolve `#attr.tessera.*` placeholders** → constants early, enabling inlining (requested earlier).

4. **Autotuning integration**  
   - Meta‑schedule surface in IR; SQLite schedule‑key cache; Hyperband search option; offline replay in CI.

---

## C) Runtime

1. **TMA & descriptor cache**  
   - Create and cache TMA descriptors once per launch; expose metrics (issue rate, stalls).

2. **FFT wrapper**  
   - cuFFT plan cache + lightweight API: `fft_conv1d(x,k,mode="same")`; fall back to tiled direct conv when small.

3. **Collectives & I/O**  
   - NVLink/IB collectives pluggable layer for multi‑GPU attention/SSM.  
   - Memory & PCIe I/O micro‑tests (bandwidth/latency) to baseline node characteristics.

4. **Observability**  
   - NVTX ranges in all library kernels; optional debug mode that records pipeline occupancy and smem usage.

---

## D) Roadmap (90‑day sketch)

### Milestone 1 (Weeks 1–4)
- Implement epilogue chaining, row‑softmax helpers, and triangular‑only masking.  
- Add WGMMA + CTA‑pair codegen for bf16; land TMA 2D op + descriptor cache.

**Acceptance:** TK GEMM + FlashAttention fwd ports within ±5% of TK on H100/B200 for target sizes.

### Milestone 2 (Weeks 5–8)
- Attention recognizer + epilogue fusion; linear‑attention convenience; selective‑scan helper.  
- Autotuner v1 (SQLite cache + schedule‑key + Hyperband).

**Acceptance:** FlashAttention bwd, linear attn, Mamba scan within ±10%; autotuner reduces hand‑tuning time by >50%.

### Milestone 3 (Weeks 9–12)
- FFT wrapper + plan cache; multi‑GPU collectives API; full CI with microbench HTML aggregation.

**Acceptance:** Long‑conv ports run within 1.1× of cuFFT‑based baselines; weekly CI reports published across sm80/sm90/Blackwell.

---

## E) Implementation notes

- Keep all helpers **expressible in raw Tile‑IR** to avoid over‑specialization.  
- Guard Blackwell‑only paths behind target feature checks; keep Hopper fallbacks.  
- Add thorough unit tests for numerics (softmax stability, FP8 scaling, dropout determinism).

<!-- MERGE_END: Tessera_Enhancement_Proposals -->
