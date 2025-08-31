# Tessera Programming Guide  
## Appendix A: NVL72 Programming Guide (Extended)

NVIDIA’s NVL72 is a 72-GPU NVSwitch domain built from GB200 superchips. Tessera programs can map directly to this hardware by treating the rack as a single, giant mesh.

---

### A.1–A.9 [as before in previous NVL72 Appendix]

---

### A.10 Mapper Recipes for NVL72

Tessera exposes a **Mapper API** to override placement, variant choice, and collective strategy.

```python
class NVL72Mapper(tessera.Mapper):
    def place(self, region, mesh):
        if region.axis == "tp":
            return "local_switch"   # co-locate TP ranks on same NVSwitch group
    def choose_collective(self, kind, size):
        return "sharp_ring"        # prefer SHARP-enabled collectives
    def choose_variant(self, op, arch):
        return "tile_ir" if arch.is_blackwell else "ptx"

tessera.runtime.set_mapper(NVL72Mapper())
```

This ensures optimal performance on NVL72: TP shards stay close, collectives exploit SHARP, and kernels use CUDA Tile IR.

---

### A.11 Index-Launch Walkthrough

Index launches simplify launching kernels across many shards.

#### Example: TP GEMM across 72 GPUs

```python
@kernel
def tp_gemm(A: f16[M,K/tp], B: f16[K/tp,N], C: mut f32[M,N/tp]): ...

# Launch kernel across TP axis (9-way tensor parallelism)
tessera.index_launch(axis="tp")(tp_gemm)(A.parts("tp"), B.parts("tp"), C.parts("tp"))
```

The runtime:  
- Launches `tp_gemm` on all 9 TP shards per stage.  
- Inserts `reduce_scatter` and `all_gather` as needed.  
- Uses NCCL over NVSwitch with SHARP reductions.  
- Control replication ensures low-latency launch across all 72 ranks.

---

### A.12 Summary

- NVL72 can be programmed as a single 72-GPU mesh.  
- Use `ShardSpec` + **domains/distributions** to express tensor layouts.  
- Collectives automatically use NVSwitch + SHARP.  
- Custom **mappers** give control over placement, collectives, and kernel variants.  
- **Index launches** provide scalable kernel distribution across shards.
