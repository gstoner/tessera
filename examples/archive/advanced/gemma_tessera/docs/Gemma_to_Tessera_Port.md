<<<MERGE BEGIN: GEMMA_TO_TESSERA_PORT >>>
# Gemma → Tessera Port Plan (Starter)

This document outlines a pragmatic path to port **Gemma** (Gemma 1/2/3-class models) to the **Tessera Programming Model**.

## Scope
- Decoder-only Transformer with RoPE, RMSNorm, SwiGLU.
- GQA/MQA attention (num_kv_heads ≤ num_heads).
- KV-cache, sampling (top-k/p, temperature), and minimal tokenizer glue.

## Mapping (Gemma → Tessera)
| Component | Gemma (JAX/PyTorch) | Tessera Primitive |
|---|---|---|
| Token Embedding | `nn.Embedding` | `tessera.tensor.lookup` |
| RoPE | rotary on q/k | `tessera.rope.apply` |
| Attention (GQA) | SDPA / FlashAttn | `tessera.attention.flash` (tile=64x64x16) |
| Norm | RMSNorm | `tessera.norm.rms` |
| MLP | SwiGLU (2*ffn → split) | `tessera.mlp.swi_glu` |
| Output proj | Linear | `tessera.mma.proj` |

## IR & Lowering
1. **High-level**: `tessera.region` per decoder block; attach tile shapes + dtypes.
2. **Mid-level**: fuse QK^T + softmax + PV into `tessera.attention.flash` with causal mask.
3. **Target IR**: lower to GPU backends (Tile IR/NVVM/ROCm) with appropriate TMA/async copies.

## Incremental Milestones
- M0: Torch baseline passes tests (done in this starter).
- M1: Replace SDPA with Tessera FlashAttention kernel (TSR path).
- M2: Add KV-cache + paged attention.
- M3: Export/Import checkpoints via converter.
- M4: Full Target IR path + microbench.

## Notes
- Weights must be fetched via official gates (HF/Kaggle) under Gemma terms.
- Keep configs aligned to released checkpoints (2B/7B/…).
<<<MERGE END: GEMMA_TO_TESSERA_PORT >>>


## Update: Native Tessera Attention + Autotune Stub
- Added `tessera_gemma/kernels/native_attention_tessera.py` (tile-blocked, online-softmax).
- `TesseraAttention` autotunes block size on first use and caches per (S,D,device,dtype).
- Forward paths now include shape-typed checks via `utils/shapes.py`.


## Update v3: KV Cache + Paged Attention + Microbench
- Added `kernels/kv_cache_tessera.py` (paged KV cache).
- Added `kernels/native_attention_paged_tessera.py` using streaming online-softmax over KV pages.
- `TesseraAttention.forward(..., kv_cache, use_cache=True)` enables paged path.
- Microbench: `scripts/bench_attention_blocks.py` prints timing across block sizes and head dims.
