# Tessera IR Layer 2 — Schedule IR (Fusion, Tiling, Sharding & Autotuning)
*(CUDA-style programming guide companion; normative unless stated otherwise)*

---

## 1. Scope
Schedule IR (**Sched IR**) transforms Graph IR into an **executable plan**:
- **Fusion**: group ops into kernels
- **Tiling**: choose tile shapes (BM, BN, BK, etc.)
- **Sharding**: map tensors to mesh axes (tp/dp/pp/ep)
- **Collectives**: insert/plan all-reduce, all-gather, reduce-scatter
- **Pipelining/Streams**: compute/comm overlap

Sched IR carries **cost annotations** and integrates an **autotuner**.

---

## 2. Entities
- **sched.func**: fusion group (future kernel)
- **sched.tile**: tile choices and loop nests
- **sched.layout**: `ShardSpec` per tensor
- **sched.collective**: NCCL/ROCm collective plan with topology hints
- **sched.pipeline**: microbatch wavefront across pp stages

---

## 3. MLIR Example (custom `tsched` dialect)
### 3.1 Fusing Attention
```mlir
tsched.module {
  %attn = tsched.fuse @attention(%Q, %K, %V)
           { pattern = "flash_attention" }

  // Shard layouts: heads on tp, batch on dp
  tsched.layout %Q {partition = ["batch"], mesh_axes = ["dp"]}
  tsched.layout %K {partition = ["head"],  mesh_axes = ["tp"]}
  tsched.layout %V {partition = ["head"],  mesh_axes = ["tp"]}

  // Tile policy candidates (BM, BN, BK)
  tsched.tile %attn candidates = [[128,128,64], [64,128,64], [128,64,64]]

  // Collective plan for logits RS/AG
  tsched.collective %attn { pattern = "RS_MM_AG", axis = "tp" }

  // Emit a sched.func for each fusion group
  %f = tsched.materialize %attn : !tsched.func
  tsched.return %f
}
```

### 3.2 Pipeline Schedule
```mlir
// Three pipeline stages across axis pp
tsched.pipeline @train { axis="pp", stages=3, microbatch=8 }
```

---

## 4. Autotuner (Normative)
### 4.1 Goals
- Pick **tile shapes**, **fusion boundaries**, **collective algos**, and **layouts** that minimize step time.
- Use both **cost models** and **on-device measurements**.
- Persist best choices **per (op, shape, dtype, mesh, arch)** signature.

### 4.2 Architecture
- **Static cost model**: predicts FLOPs, bytes (HBM/NVLink), occupancy → time estimate.
- **Dynamic runner**: measures kernel/collective latencies on target GPUs.
- **Cache**: persistent DB (e.g., sqlite) keyed by signature:
  `key = (op_kind, shape_sig, dtype, mesh_sig, arch, policy_hash)`
- **Warm-start**: load vendor profiles (GB200 defaults) and prior runs.

### 4.3 API (Pseudo-Python)
```python
from tessera import autotune

sig = autotune.signature(op="flash_attention", shape=(B,H,L,D), dtype="fp8_e4m3",
                         mesh={"tp":12,"dp":1}, arch="GB200", policy="stable" )

plan = autotune.plan(
    candidates=dict(
        tiles=[(128,128,64),(64,128,64),(128,64,64)],
        rsmmag=[True, False],
        a2a_quant=["off","fp8"],
        fuse_bias=[True, False]
    ),
    objectives=["latency","bytes_moved"],
    constraints={"HBM_util": "<0.85", "deterministic": True},
    signature=sig
)

best = autotune.search(plan, mode="hybrid")  # cost model + measurements
autotune.cache.store(best)
```

### 4.4 MLIR Hooks
```mlir
// Attach autotune results to sched IR
tsched.tile %attn chosen = [128,128,64] { source = "autotune" }
tsched.collective %attn { algo = "ring", quantize = "fp8" }
```

### 4.5 Measurement Protocol
- **Warm-up** 3 runs, **measure** ≥ 10 runs, report median.
- Capture **power**, **HBM BW**, **NVLink BW**, **occupancy** (optional).
- Plans that violate determinism or memory caps MUST be discarded.

### 4.6 Persistence & Invalidation
- Entries expire on arch/driver/runtime hash change.
- Miss → re-tune only the affected fusion groups.
- Export/import tuning DB for reproducibility.

---

## 5. Determinism
- Autotuner MUST NOT change numeric semantics (e.g., reduction order).
- Only layout/tile/overlap/quantization-within-policy are tunable.

---

## 6. Validation
- Compare tuned vs. baseline with confidence threshold (e.g., ≥7% speedup to accept).
- Keep a **safe plan** fallback for portability.
