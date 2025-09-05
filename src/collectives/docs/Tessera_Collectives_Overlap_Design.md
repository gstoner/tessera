# Tessera Collectives: Overlap-First Design (Draft)

**Version:** v1  
**Status:** Draft for review  
**Scope:** Programming model, runtime scheduler, autotuning knobs, and validation plan for communication–compute overlap using chunked collectives.

---

## 1. Goals

- Treat **collectives as first-class, schedulable work** (like kernels/tiles).
- **Pipeline and overlap** collectives with compute by construction.
- Auto-select **algorithm** and **transport** per message size + topology.
- Enable safe **bandwidth reduction** (precision/quantization) with guardrails.
- Provide **portable IR** + minimal runtime changes to unlock benefits day one.

---

## 2. Programming Model Changes

### 2.1 Futures and true dependencies
- Every collective returns a **future** handle.
- Compute blocks **await()** only at **true use sites** (no global barriers).
- Example:
  ```mlir
  %f = tessera.collective.reduce_scatter %grad : memref<...> {op="sum"}
  // ... do other compute here ...
  %shard = tessera.await %f  // only where actually needed
  ```

### 2.2 Collective scopes and subgroups
- `scope = {intra_sm, intra_gpu, node, rack}` hints the hierarchy.
- `subgroup = !tessera.shard<...>` describes the logical shard the op acts on.
- Facilitates composition of **warp→block→device→NCCL/RCCL** tiers.

### 2.3 Chunking and pipelining
- `chunk_bytes` (or rows/cols) slices tensors into **tile-aligned chunks**.
- Enables **double/triple-buffered** pipelines in fwd/bwd.
- Example, BWD overlap:
  ```mlir
  %dw_k     = tessera.compute.matmul %a_k, %b_k
  %ar_fut   = tessera.collective.all_reduce %dw_k_prev {chunk=512KiB}
  tessera.store %dw_k -> @stash
  %dw_km1  = tessera.await %ar_fut
  ```

### 2.4 Fusion policy
- Concatenate small tensors to one collective if consumer can split.
- Fold **scale/cast** into the collective (reduce in FP32, transmit BF16/FP8).

### 2.5 Precision and compression (safe defaults)
- `dtype=bf16|fp16|fp8`, `qscheme=blockwise`, **reduce in FP32**.
- Optional **error-feedback** for compressed gradients.
- Optional **top‑k sparsity** (indices + values) for large grad tensors.

---

## 3. Algorithm & Transport Selection

### 3.1 Algorithm policy (auto)
- **Ring** for large messages (bandwidth-optimal).
- **Tree / recursive halving/doubling** for mid sizes (latency-lean).
- **Hierarchical** (NVLink/NVL intra-node + RDMA inter-node).
- **Direct-send** for tiny messages.

### 3.2 Transport policy (auto)
- Prefer **NVLink/NVL** intra-node; **PCIe** fallback.
- Inter-node use **GPUDirect RDMA** if available.
- Expose as attributes, e.g. `algo="auto"`, `path="auto"`, with env overrides.

### 3.3 Topology descriptor
- Runtime maintains a graph with **nodes (GPU/NIC/switch)** and **edges (BW, latency)**.
- Compiler queries it to sanity-check chunk sizes and inflight depth.

---

## 4. Runtime & Scheduler

### 4.1 Two-queue model per device
- **ComputeQ**: tiles/kernels.  
- **CommQ**: chunked collective work items.  
- Rule of thumb: keep `CommQ` ≥ *N* inflight chunks while `ComputeQ` has ready tiles.

### 4.2 Credit-based link scheduler
- Maintain credits per **NVLink lane** and **NIC queue**.  
- Do not issue chunks that would exceed credits; pick another ready unit.

### 4.3 Progress engine
- Lightweight CPU or resident GPU thread advancing non-blocking ops and completing futures.

### 4.4 Hierarchical collectives (composed automatically)
- Stage 1: warp/block reduction → Stage 2: intra-node over NVLink/NVL → Stage 3: inter-node RDMA tree.  
- Surfaces as a single op to the user.

---

## 5. Autotuning

### 5.1 Knobs
- `chunk_bytes`, `max_inflight`, `algo`, `path`, `pipeline_depth`, `compress`.

### 5.2 Cost model (per chunk)
- Latency ≈ `α * hops + n / B_effective`.
- **Overlap score:** penalize frames where ComputeQ idles while CommQ is busy (or vice versa).

### 5.3 Online adaptation
- Adjust `chunk_bytes` and `max_inflight` using moving percentiles of measured BW/latency; pin once stable.

---

## 6. Instrumentation & Validation

- **NVTX/Nsight/Perfetto hooks** per chunk; overlay **communication rooflines** next to compute rooflines.
- **Correctness suite:** randomized tensor shapes/dtypes/shard layouts; compression roundtrip checks; error‑feedback invariants.

---

## 7. Canonical Pipelines

### 7.1 Data-parallel training (RS/AG)
- BWD: **reduce_scatter** grads layer-by-layer, `chunk=512KiB–2MiB`, `max_inflight=4–8`.
- FWD: **all_gather** weights just-in-time per layer.

### 7.2 MoE (A2A)
- Bucket by expert, issue **all_to_all** per bucket; fuse pack/cast with send.

### 7.3 Param prefetch
- Start `all_gather(W_{k+1})` when layer *k* compute begins; `await()` right before use.

---

## 8. IR Checklist (repo landing)

- New ops returning **futures** with attrs: `{chunk, algo, path, scope, dtype, priority, overlap, qos}`.
- Planner pass: **chunk slicing** + **await insertion** only at true use sites.
- NCCL/RCCL adapters supporting chunked submissions + callbacks.
- Autotuner + telemetry; size→algo LUT with env overrides.
- Lit tests for **IR→plan** and a small Perfetto export sample.

---

## 9. Appendix: Minimal Examples

### 9.1 All-reduce with overlap
```mlir
%dw      = tessera.compute.matmul %x, %y
%f       = tessera.collective.all_reduce %dw 
            {op="sum", chunk=1MiB, dtype="bf16", algo="auto", path="auto", scope="node"}
; ... compute next tile ...
%dw_red  = tessera.await %f
%out     = tessera.apply_epilogue %dw_red {scale = %eta}
```

### 9.2 Reduce-scatter / All-gather pair
```mlir
%f_rs  = tessera.collective.reduce_scatter %grads {op="sum", chunk=2MiB}
%shard = tessera.await %f_rs
; use shard locally ...
%f_ag  = tessera.collective.all_gather %weights_sharded {chunk=2MiB}
%W     = tessera.await %f_ag
```
