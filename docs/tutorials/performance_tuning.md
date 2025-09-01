# Tessera Performance Tuning Guide
**File:** `performance_tuning.md`  
**Version:** 1.0 (2025‑09‑01)  
**Audience:** Kernel authors, model integrators, and systems engineers using the Tessera Programming Model.

---

## 0. TL;DR Checklist (Start Here)
- [ ] Define the **unit of work** per tile (elements/outputs produced per tile).
- [ ] Compute **arithmetic intensity** (FLOPs/byte) and place the kernel on the Roofline.
- [ ] Pick a **tile shape** that matches tensor core fragment sizes and SM resources.
- [ ] Ensure **coalesced global loads/stores** (128–256B segments); align pointers.
- [ ] Stage through **shared memory** with **double/triple buffering** and `async` copies.
- [ ] **Pad/permute shared memory** to avoid bank conflicts (32/64B stride padding).
- [ ] Use **warp specialization** or **pipeline stages** to overlap copy/compute.
- [ ] Keep **register pressure** below the spill threshold; verify **occupancy** sufficiency.
- [ ] Add **NVTX ranges** (`"copy"`, `"mma"`, `"reduce"`, `"comm"`) for profiling.
- [ ] Validate correctness at small sizes; then sweep **problem sizes**, **tile sizes**, and **precision**.
- [ ] For multi‑GPU: choose **bucket sizes** to overlap **compute/collectives**; verify **topology**.
- [ ] Log metrics (CSV/JSONL); track **p95 latency**, **GB/s**, **TFLOP/s**, **efficiency**.

---

## 1. Performance Model: Where Time Goes
Let total step time be:
\[
T \approx \max\big(T_{\text{compute}},\ T_{\text{mem}},\ T_{\text{sync}},\ T_{\text{comm}}\big) + T_{\text{launch}} + T_{\text{host}}
\]
- \(T_{\text{compute}}\): tensor‑core MMAs, ALU.
- \(T_{\text{mem}}\): HBM/L2/L1/SMEM traffic; coalescing & reuse decide this term.
- \(T_{\text{sync}}\): barriers, reductions, atomics, warp divergence.
- \(T_{\text{comm}}\): NVLink/NVL/PCIe/NIC collectives, host‑device copies.
- \(T_{\text{launch}}, T_{\text{host}}\): kernel launch, CPU orchestration.

**Arithmetic intensity:** \(I = \frac{\text{FLOPs}}{\text{Bytes from HBM}}\).  
If \(I < \frac{P_{\text{peak}}}{B_{\text{HBM}}}\) → memory‑bound; increase reuse, fuse ops, shrink bytes.  
If \(I \ge \frac{P_{\text{peak}}}{B_{\text{HBM}}}\) → compute‑bound; raise **tensor‑core occupancy** and pipeline depth.

---

## 2. Single‑GPU Kernel Tuning

### 2.1 Tile Shape & Threadblock Layout
- Choose \((M, N, K)\) to map onto MMA fragments (e.g., 16×16×16 like shapes) and SMEM.
- Target **2–4 stages** of pipeline depth for \(K\)-slices; consider **warp specialization**: *loaders* vs *computers*.
- Keep **active warps per SM** high enough to hide latency but not so high that register spills occur.
- Heuristics:
  - Start with **square-ish output tiles** (e.g., 128×128) and step down if spills/SMEM overflows.
  - Prefer **K‑major traversal** with **prefetch distance** of 2–3 tiles.

### 2.2 Memory Movement
- **Global ↦ SMEM:** use asynchronous copies; group/commit/wait to overlap with MMAs.
- **Coalescing:** 32/64/128‑thread wide vectorized loads; align to 128–256B segments.
- **Reuse:** keep *A/B* panels in SMEM; accumulate in registers; write once.
- **L2 residency:** block re‑order to increase temporal locality; avoid thrashing by too many concurrent CTAs touching the same cache sets.
- **Stores:** avoid partial writes; prefer `st.global.cs`‑like commit patterns (write‑once, streaming).

### 2.3 Shared Memory: Conflicts & Layout
- **Bank conflicts:** pad leading dimension by +1/2 elements so that stride ≠ multiple of bank width.
- **Swizzle/permutation:** interleave rows/columns to avoid hot banks during MMA.
- **Atomic reductions:** aggregate per‑warp into SMEM then single atomic to global to reduce contention.

### 2.4 Tensor Cores & Numerics
- **Precision:** FP16/BF16 inputs with FP32 accumulators for training; FP8 where accuracy allows.
- **Tile MMA pipeline:** interleave `mma.sync` with `ldmatrix`/prefetch to keep pipes full.
- **Fused epilogues:** bias, activation, scaling fused into the epilogue to avoid extra HBM traffic.

### 2.5 Control Flow & Divergence
- Replace branches with **predication** where possible.
- Structure tiles so conditional masks are **per‑tile** or **per‑warp** (reduce intra‑warp divergence).

### 2.6 Latency Hiding Patterns
- **Double buffer** A/B panels: while computing on \(k\), prefetch \(k+1\).
- **Triple buffer** if memory latency is dominant and registers/SMEM allow.
- **Warp specialization:** dedicate one warp to `async_copy`, others to MMAs; hand‑off via barriers.

### 2.7 Occupancy & Resource Balancing
- Watch **registers/thread** and **SMEM/CTA**; reduce unroll or split epilogues if spilling occurs.
- Aim for **≥ 2 CTAs/SM** for latency hiding (heuristic; depends on kernel). Validate with profiler.

### 2.8 Common Performance Smells → Fixes
| Smell | Likely Cause | Quick Fixes |
|---|---|---|
| Low tensor‑core utilization | Insufficient pipeline depth | Add stages; overlap load/compute; use warp specialization |
| High L2 misses | Irregular access / too many CTAs touching same sets | Re‑tile; reorder blocks; increase reuse in SMEM |
| Shared memory replays | Bank conflicts | Pad/permute SMEM layout |
| Register spills | Over‑unrolling / too large tile | Reduce tile, split epilogue, limit live ranges |
| Poor store efficiency | Partial/strided stores | Switch to row‑major epilogue; accumulate then write once |
| Sync stalls | Overuse of barriers | Use finer scopes (warp‑level); pipeline instead of global sync |

---

## 3. Multi‑GPU & Scale‑Out

### 3.1 Topology & Collectives
- Detect **NVLink/NVL** vs **PCIe** topology; choose **ring** for large payloads, **tree** for small/latency‑sensitive ops.
- **Bucket sizes:** 8–32 MB starting point; sweep by bandwidth/latency product.
- **Overlap:** chunk gradients/activations; launch comm on a dedicated stream while compute proceeds.

### 3.2 Parallelism Modes
- **Data parallel:** maximize batch; gradient bucketing + overlap.
- **Tensor/model parallel:** shard large GEMMs; fuse intra‑layer comm; align shard sizes to tile boundaries.
- **Pipeline parallel:** micro‑batch to fill the pipe; balance stage times; mitigate bubbles.

### 3.3 Host/NUMA & I/O
- Pin host threads to NUMA node attached to the GPU.
- Use **pinned memory** for H2D/D2H; overlap with compute streams.
- Avoid cross‑socket traffic for staging buffers when possible.

---

## 4. End‑to‑End Step Tuning

### 4.1 Five‑Pass Method
1. **Correctness**: tiny cases, deterministic seed, assert invariants.
2. **Roofline**: compute \(I\); estimate bound; set performance target.
3. **Memory path**: coalescing, async copies, SMEM conflicts, write‑once epilogue.
4. **Compute path**: MMA utilization, pipeline depth, unroll factors.
5. **Overlap**: fuse epilogues, interleave comm/compute, hide H2D/D2H.

### 4.2 Logging & Reproducibility
- Log CSV: `ts, step, kernel, tile, dtype, flops, bytes, t_ms, tput, eff, p95`.
- **NVTX ranges** around major phases to correlate timelines.
- Fix **Philox seed** for deterministic micro‑benchmarks; record env/hardware hash.

---

## 5. Tessera‑Specific Recipes

### 5.1 TileLinear (GEMM‑like)
- Start with 128×128×K tiles; 2–3 pipeline stages.
- Use `async_copy` to bring A/B panels; `mma` in inner loop; fused bias+activation in epilogue.
- Check: tensor‑core active ≥ 80%, dram throughput near roofline if memory‑bound.

### 5.2 FlashAttention (Fused QKᵀ → Softmax → V)
- Block along **sequence** so that Q/K tiles fit in SMEM; stream V by chunks.
- Apply **causal/mask** via predicates to avoid divergence.
- Fuse **softmax scaling**, **dropout**, and **V matmul** in the same kernel to minimize HBM IO.
- Backward: accumulate dV/dK/dQ in registers; stage partials in SMEM to reduce atomics.

### 5.3 LayerNorm / RMSNorm
- Use warp‑wide reductions with **shfl** ops; SMEM only for cross‑warp combine.
- Fuse scale/bias and optional activation; vectorize loads/stores (e.g., 128‑bit).

### 5.4 Stencil / PDE Tile
- Prefer read‑only cache for coefficients; halo exchange via SMEM with padding.
- Avoid warp divergence on boundary conditions by masking at tile granularity.

---

## 6. Validation & Micro‑Benches
- **Kernel sweeps:** problem sizes, tile shapes, dtype (FP16/BF16/FP8), pipeline stages.
- **Stress tests:** sustained load (minutes) to reveal thermal throttling/clock drift.
- **A/B tests:** isolated kernel vs fused version; measure HBM bytes and time separately.
- **Regression gates:** ±2–3% thresholds on TFLOP/s and GB/s with fixed seeds.

---

## 7. Metric Cheat‑Sheet (Profiler‑Oriented)
*(Names vary by tool; treat below as representative)*
- **Tensor core active**: time % with MMA pipes busy.
- **DRAM throughput**: GB/s relative to peak; read/write balance.
- **L2 hit rate**: % hits; evictions.
- **SMEM bank conflicts / replays**: replay factor near 1.0 is ideal.
- **Eligible warps per cycle**: proxy for latency hiding.
- **Warp stall reasons**: memory dependency, barrier, scoreboard, not selected.
- **Occupancy**: active warps / max warps; also CTAs/SM.
- **Branch efficiency**: near 100% for uniform control flow.

---

## 8. Tuning Playbooks

### 8.1 If Memory‑Bound
- Increase tile reuse (larger \(K\) slice or fuse ops).
- Vectorize IO (128/256‑bit).
- Stream once; avoid read‑modify‑write of large tensors.

### 8.2 If Compute‑Bound
- Deepen the pipeline (more stages).
- Raise math density (fuse epilogues, accumulate more work per tile).
- Ensure fragments align with tensor core shapes; unroll inner loops.

### 8.3 If Sync‑Bound
- Replace global barriers with warp‑scope syncs.
- Aggregate reductions hierarchically (lane→warp→CTA→grid).

### 8.4 If Comm‑Bound
- Increase bucket size; overlap with compute; switch ring↔tree based on payload.
- Compress (quantize) gradients/activations if accuracy allows.

---

## 9. Performance Bug Report Template
```
Title: <short summary>
Hardware/Topology: <GPU model, links, NICs>
Software: <driver, runtime, commit hash>
Kernel/Step: <name>
Problem Size: <dims>
Tile Config: <M,N,K, stages, warps>
dtype: <fp16/bf16/fp8>
Metrics: <TFLOP/s, GB/s, p95, occupancy, stalls>
Timeline: <key NVTX ranges with times>
Repro: <cmd + seed>
```

---

## 10. Glossary
- **Tile**: Unit of computation mapped to a threadblock/warp group.
- **Pipeline stage**: Copy/compute phase overlapped in time.
- **Epilogue**: Post‑accumulation compute (bias, activation, scale) before store.
- **Bucket**: Message chunk for collective communication.

---

## 11. Cross‑References (Tessera Docs)
- `Tessera_Programming_Model_V1.md` — conceptual overview.
- `tessera_frontend_architecture.md` — language & frontend.
- `tessera_target_ir_complete_unified.md` — IR & lowering passes.
- `tessera_target_ir_usage_guide.md` — IR usage examples.
- `Tessera_Standard_Operations.md` — canonical ops and epilogues.
- `system_overview.md` — runtime/system integration.

> Informative: The above internal docs complement this guide; sections are aligned to the tile‑first, IR‑lowering approach in Tessera.

---

## 12. Appendix: Example Kernel Skeleton (Pseudo‑Tessera)
```tessera
kernel tile_linear(A, B, C, bias, M, N, K) {
  // Tile params (tune these)
  let TM = 128, TN = 128, TK = 64;
  let stages = 3;

  // Shared memory panels (padded to avoid bank conflicts)
  smem A_panel[TM][TK + 1];
  smem B_panel[TK][TN + 1];

  // Acc registers
  reg acc[TM_frag][TN_frag] = 0;

  // Prologue: prefetch first stages
  for s in 0..stages-1 {
    async_copy(A_panel[s], A[... offset(s) ...]);
    async_copy(B_panel[s], B[... offset(s) ...]);
    async_commit();
  }

  barrier();

  for k in 0..K step TK {
    // Issue next stage prefetch while computing current
    async_copy(A_panel[next], A[... offset(k + stages*TK) ...]);
    async_copy(B_panel[next], B[... offset(k + stages*TK) ...]);
    async_commit();

    // Compute: ldmatrix + mma.sync
    #pragma unroll
    for kk in 0..TK step frag_k {
      frag a = ldmatrix(A_panel[cur], kk);
      frag b = ldmatrix(B_panel[cur], kk);
      acc = mma(acc, a, b);
    }

    async_wait_group(1);
    swap(cur, next);
  }

  // Fused epilogue: bias + activation
  store_streaming(C, fuse(acc, bias));
}
```

---

**End of Guide**
