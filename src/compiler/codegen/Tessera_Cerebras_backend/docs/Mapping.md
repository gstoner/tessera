# Tessera ↔ Cerebras mapping (condensed)

- **Tessera tiles / tile-groups** → **Rectangular PE regions** described by `SdkLayout`-style JSON.
- **Tessera memspaces**: 
  - `mem.tile` → PE SRAM (observe banked RW constraints).
  - `mem.global` → host tensors transferred via SDK memcpy streams.
  - **Weights**: pipeline mode (on-wafer) vs **weight streaming** (MemoryX/SwarmX).
- **Tessera scheduling/barriers** → CSL control + phase boundaries; host orchestrates multi-stage pipelines.
- **Runtime** → `SdkRuntime` for program load, param binding, tensor streaming, and launch.

## First kernels to bring up

1) **GEMV/GEMM**: rectangular partition the output tile-space across regions.  
2) **LayerNorm / reductions**: single-region kernels, multi-region reduce when necessary.  
3) **FlashAttention (tiny)**: stream Q/K/V blocks from host; keep softmax scratch in SRAM; optional weight streaming for large models.
