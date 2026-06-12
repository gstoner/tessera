---
status: Tutorial
classification: Tutorial
last_updated: 2026-06-11
---

> **Phase status note (updated 2026-06-11):** Phases 1–7 are complete and Phase 8 (Apple M-Series CPU via Accelerate, GPU via Metal/MPS/MPSGraph/custom MSL) is operational — on Apple Silicon this is the primary single-node execution path. Autodiff (forward/reverse transforms + activation checkpointing), ZeRO-2 optimizer sharding, the Bayesian autotuner, and the runtime Python wrapper (`tessera.runtime.TesseraRuntime`) are **shipped**. Genuinely still planned: **multi-GPU / multi-rank** execution of distributed collectives (NCCL/RCCL), `Cyclic` distribution lowering, and **NVL72** rack-scale execution (single-device collectives run over in-process mock ranks today). Canonical API names: `docs/CANONICAL_API.md`; phase table: root `CLAUDE.md`.


# Tessera Programming Guide  
## Appendix A: NVL72 Programming Guide (Extended)

> ⚠️ **Entirely Phase 4 planned — not operational.** Every example in this
> appendix describes Tessera's *intended* rack-scale execution on a 72-GPU
> NVSwitch domain. None of it runs today: multi-GPU/multi-rank execution,
> NVSwitch+SHARP collectives, and custom mapper policies are all future work,
> and the dev environment has no such hardware. For **executable** examples
> use `@tessera.jit(target="apple_gpu")` / `"apple_cpu"` / x86 (see Chapters
> 2–5 and 10). This appendix is a design/roadmap reference.

NVIDIA’s NVL72 is a 72-GPU NVSwitch domain built from GB200 superchips. Tessera’s NVL72 support is Phase 4 planned. This appendix is a future-facing guide, not current API guidance.

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
