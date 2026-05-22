<!-- MERGE_START:DINOv3_Tessera_Spec -->
# DINOv3 → Tessera Port Notes (v2)

This revision replaces more PyTorch modules with **Tessera-style tile ops** and
adds a **reference FlashAttention** implemented as a streaming log-sum-exp across K/V blocks.

## What changed in v2

- **TileLinear** replaces nn.Linear in QKV, attention proj, and MLP layers.
- **tessera_layer_norm** with fused affine is used throughout (two LN sites per block + final LN).
- **Reference FlashAttention** in tiles: streaming per-row m/l accumulators; dropout supported.
- **Token-level Gram anchoring**: computes Grams on **patch tokens** at selected transformer depths.
- Configurable `gram_layers` (e.g., `[6, 12]`), routed end-to-end.
- Environment flags:
  - `TESSERA_USE_CUSTOM_KERNELS=1` → load your real Tessera kernels.
  - `TESSERA_REFERENCE_KERNELS=0` → use PyTorch SDPA when custom kernels are unavailable.

## Kernel boundaries

| Layer | Replaced with Tessera op | Notes |
|---|---|---|
| QKV linear | `TileLinear` | Prepare for fused QKV in-kernel later |
| Attention | `tessera_flash_attn` | Streaming softmax; block size from `TileSchedule.block_n` |
| Output proj | `TileLinear` | Projection back to model dim |
| MLP fc1/fc2 | `TileLinear` | fc1 has fused GELU |
| LayerNorm | `tessera_layer_norm` | Fused affine; chunked reductions |

## Gram anchoring details

- For each selected layer `L`, we collect **patch tokens** (exclude `cls`) and compute
  `G = (Xᵀ X)/(B·N·C)` averaged over batch and tokens; student Grams are matched to the
  average teacher Gram over the two global crops using MSE.

## Next steps for true Tessera kernels

- Swap `TileLinear` inner loops with a real **tile GEMM** (shared memory + barriers).
- Replace `_reference_flash_attention` with your Tessera `flash_attn` kernel.
- Add **tile-local reductions** for accurate Gram computation per block if desired.
<!-- MERGE_END:DINOv3_Tessera_Spec -->
