# DINOv3 → Tessera (All-in-One Package)

**Build date:** 2025-09-01T19:12:00

This package includes the full tessera-optimized DINOv3 port with CUDA kernels and training support.

## Highlights
- TileLinear forward/backward (CUDA) with **WMMA FP16/BF16** fast paths (accum FP32)
- LayerNorm forward/backward (CUDA)
- FlashAttention fused forward/backward with **causal + dropout** and deterministic seeding
- **Batched GEMM** (tiled + HMMA WMMA variants) and **rowwise softmax**
- **Fused QKV single-GEMM pack** with fused backward (returns dX/dWcat/dbcat; GELU′ folded into Q)
- ViT attention wired to the fused QKV pack behind a flag (default **ON**)

## Quickstart
```bash
pip install -r requirements.txt
python setup.py build_ext --inplace    # or: pip install -v .

export TESSERA_USE_CUSTOM_KERNELS=1
# default is on; set to 0 to disable the single-GEMM QKV pack
export TESSERA_USE_QKV_PACK=1

# sanity train
python -m examples.train_dinov3 --fake-data --steps 10

# benches
python -m bench.bench_attention
python -m bench.bench_qkv_pack
python -m bench.bench_training
```

## Flags
- `TESSERA_USE_CUSTOM_KERNELS=1` — enable CUDA kernels
- `TESSERA_USE_QKV_PACK=1|0` — fused single-GEMM QKV pack on/off (default on)
- `TESSERA_NAIVE_ATTENTION=1` — force naive attention (no torch.matmul; uses our kernels)
- `TESSERA_REFERENCE_KERNELS=1` — use reference streaming kernel for checks
- `dropout_p`, `causal`, `seed` — knobs on `tessera_flash_attn(...)`

## Notes
- WMMA paths require dims multiple of 16 and FP16/BF16 tensors; accumulation is FP32.
- For BF16 WMMA backward in TileLinear, autograd routes to tensor-core paths automatically when eligible.
