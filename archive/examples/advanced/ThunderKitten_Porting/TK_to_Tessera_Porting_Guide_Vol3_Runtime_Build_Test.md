<!-- MERGE_START: TK_to_Tessera_Porting_Guide -->
# ThunderKittens → Tessera Porting Guide (Vol. 3: Runtime, Build, and Tests)

**Date:** 2025-09-02  
**Scope:** Runtime integration (TMA, pipelines, barriers), build/target knobs, autotuning, and CI/testing.

---

## 1) Target features & build knobs

- **Hopper (H100):** WMMA, TMA 2D, standard CTA scheduling.  
- **Blackwell (B100/B200):** WGMMA, CTA‑pairs, improved TMA; enable `--tessera:use_cta_pairs --tessera:use_wgmma`.  
- **Dtypes:** bf16/fp16 baseline; fp8 (e4m3/e5m2) with per‑tile scale/dequant epilogues.

Add CMake presets to toggle these per‑target and emit ptxas flags as needed.

---

## 2) Runtime integration patterns

- **Pipelines:** `pipeline.begin() → async_copy → commit → wait(slot) → compute → swap(slot)`; use two or three smem buffers.  
- **Barriers:** prefer arrive/wait paired with pipeline slots; avoid CTA‑wide `__syncthreads()` in the steady‑state when possible.  
- **Descriptors:** cache TMA descriptors per‑kernel launch to amortize setup cost.

---

## 3) Autotuner hooks

- Provide meta‑schedule knobs: `block_m, block_n, block_k, stages, split_k, cta_pairs, smem_layout`.  
- Add a **SQLite schedule cache** keyed by `arch, dtype, op, shape, strides, knobs`.  
- Integrate a **Hyperband** searcher for quick wins; fall back to exhaustive in offline mode.

---

## 4) Testing & microbench

- **Correctness harness:** tiny/odd sizes; randomized tensors; reproducible dropout via Philox seed.  
- **Perf bench:** report TFLOP/s and GB/s; emit HTML with charts from CSV (per‑arch tabs).  
- **Stress:** optional GPU‑fryer‑style stress test; NVTX ranges around load/compute/store segments.

---

## 5) CI suggestions

- Weekly jobs on multiple GPU labels (sm80/sm90/Blackwell) aggregating reports into one HTML with tabs.  
- Store schedule caches as build artifacts; compare regressions against last good run.

---

## 6) Ready‑to‑port list (Vol. 3)

- Runtime wrappers for TMA descriptor caching, pipeline helpers, autotuner interface, microbench + CI integration.

<!-- MERGE_END: TK_to_Tessera_Porting_Guide -->
