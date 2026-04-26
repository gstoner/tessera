---
status: Tutorial
classification: Tutorial
last_updated: 2026-04-26
---

> **Phase status note:** Unless this document explicitly says otherwise, distributed collectives (NCCL/RCCL), TPU StableHLO, Cyclic distribution, autodiff transforms, activation checkpointing, ZeRO sharding, Bayesian autotuning, the runtime Python wrapper, production deployment, and NVL72 execution are Phase 4-6 planned as defined in `docs/README.md`. Current Phase 1-3 API names are defined in `docs/CANONICAL_API.md`.


# Tessera Programming Guide  
## Appendix A: NVL72 Programming Guide (Extended)

NVIDIA’s NVL72 is a 72-GPU NVSwitch domain built from GB200 superchips. Tessera’s NVL72 support is Phase 4 planned. This appendix is a future-facing guide, not current Phase 1-3 API guidance.

---

### A.1–A.9 [as before in previous NVL72 Appendix]

---

### A.10 Future Mapper Recipes for NVL72

Mapper APIs are Phase 4 planned. Future mapper policy may override placement, variant choice, and collective strategy.

```python
# Phase 4 planned sketch
# mapper.place(region, mesh)
# mapper.choose_collective(kind, size)
# mapper.choose_variant(op, arch)
```

The intended behavior is to keep TP shards close, select SHARP-capable collectives where available, and choose backend variants suited to the GPU generation.

---

### A.11 Index-Launch Walkthrough

Index launches simplify launching kernels across many shards.

#### Example: TP GEMM across 72 GPUs

```python
@tessera.kernel
def tp_gemm(
    A: tessera.f16[..., ...],
    B: tessera.f16[..., ...],
    C: tessera.mut_f32[..., ...],
):
    C[:] = tessera.ops.gemm(A, B)

# Launch kernel across TP axis (9-way tensor parallelism)
tessera.index_launch(axis="tp")(tp_gemm)(A.parts("tp"), B.parts("tp"), C.parts("tp"))
```

Current Phase 1-3 behavior can test this shape with shard lists and mock execution. Phase 4 planned runtime work adds:

- Launch across all TP shards per stage.
- Inserted `reduce_scatter` and `all_gather` where required.
- NCCL over NVSwitch with SHARP reductions where available.
- Low-latency replicated launch control across ranks.

---

### A.12 Summary

- NVL72 as a single 72-GPU mesh is Phase 4 planned.  
- Use `ShardSpec` + **domains/distributions** to express tensor layouts.  
- Collectives over NVSwitch + SHARP are Phase 4 planned.  
- Custom mapper policies are Phase 4 planned.  
- **Index launches** provide scalable kernel distribution across shards.
