# DINOv3 → Tessera Port (Scaffold)

This is a working scaffold to run **DINO-style self-supervised pretraining** with a minimal ViT backbone,
wired so critical kernels (attention, layer norm, softmax) can be routed through **Tessera tile kernels**.
If Tessera kernels are not present, we **fallback to PyTorch**.

> ✅ Goal: get you to a *runnable baseline* right away, while isolating the touch points where Tessera
> tiles, shared memory, and barriers plug in.

## What’s here

- `models/dinov3_tessera/vision_transformer_tsr.py` – compact ViT with hooks for Tessera ops.
- `models/dinov3_tessera/dino_head.py` – projection head & prototypes.
- `models/dinov3_tessera/ssl.py` – DINO-style loss with **teacher EMA**, centering & temperatures.
- `models/dinov3_tessera/augment.py` – multi-crop pipeline (2× global, N× local) + color jitter, blur.
- `models/dinov3_tessera/ops/flash_attention_tessera.py` – Tessera kernel stubs + PyTorch fallback.
- `examples/train_dinov3.py` – toy run that exercises forward/backward on fake data or ImageNet-like folders.
- `configs/base.yaml` – knobs for backbone, crops, temperatures, EMA, Gram anchoring weight, etc.
- `tests/test_forward.py` – smoke test: forward pass + loss on tiny batch.

## Tessera integration points

- `ops/flash_attention_tessera.py` exposes `tessera_flash_attn(q, k, v, ...)` and `tessera_layer_norm(..)`.
- Set environment `TESSERA_USE_CUSTOM_KERNELS=1` to attempt to import real Tessera kernels:
  `from tessera_kernels import flash_attn, layer_norm`.
- Tile sizing, shared memory, and barriers are specified in the `TileSchedule` dataclass.

## DINOv3 specifics supported

- Student/Teacher with EMA & teacher centering.
- Multi-crop (2× global + K× local) with cross-view loss.
- **Gram anchoring** hook (`ssl.py::gram_anchor_loss`) – opt-in via config.
- Post-hoc high-res adaptation stub (resize positional encodings).

> Note: This scaffold focuses on *correctness & structure* rather than full-scale SOTA training runs.
> You can dial up model size and batches once Tessera kernels are in place.

## Quick start

```bash
# (Optional) Point to an ImageNet-style folder if you want real images:
export DATA_ROOT=/path/to/imagenet_like

# Try a tiny run on fake data
python -m examples.train_dinov3 --fake-data --steps 5

# Or real data with small crops
python -m examples.train_dinov3 --data $DATA_ROOT --epochs 1 --batch-size 64
```

## License

This scaffold is provided for integration/testing purposes. You are responsible for complying with the licenses of datasets and upstream models you train or evaluate.


---

## v2 highlights
- Replaced nn.Linear with **TileLinear** in backbone (QKV/proj/MLP).
- Implemented **reference FlashAttention** (streaming, block-wise) under Tessera API.
- Switched all LayerNorms to **tessera_layer_norm** (fused affine).
- Enabled **token-level Gram anchoring** via `SSLConfig.gram_layers`.
- Example script exposes `--gram-weight` and `--gram-layers`.


## Building the CUDA/Tessera kernels

```bash
# In this folder:
pip install -r requirements.txt  # if you maintain one
python setup.py build_ext --inplace   # or: pip install -v .
```

Then enable the custom kernels:
```bash
export TESSERA_USE_CUSTOM_KERNELS=1
python -m examples.train_dinov3 --fake-data --steps 10
```

## Early micro-benchmarks

Use `bench/bench_tilelinear.py` to compare **TileLinear** vs `nn.Linear`.



## New in v4: Batched GEMM, Naive Attention (no torch.matmul), Small Fused FlashAttention
- `tessera_kernels.batched_gemm(A,B,trans_b)` — drives **QK^T** and **P·V** without PyTorch matmul.
- `tessera_kernels.rowwise_softmax` — used in naive path for per-row softmax.
- `tessera_kernels.flash_attn_forward(Q,K,V)` — a simple fused attention kernel (float32, no mask), suitable for benchmarking.
- Toggle paths:
  - `TESSERA_USE_CUSTOM_KERNELS=1` — prefer CUDA kernels.
  - `TESSERA_NAIVE_ATTENTION=1` — force naive path using our kernels (no torch.matmul).
  - `TESSERA_REFERENCE_KERNELS=1` — use the streaming reference kernel instead of fused/naive.

### Attention micro-bench
```bash
export TESSERA_USE_CUSTOM_KERNELS=1
python -m bench.bench_attention   # prints ref / naive / fused timings
```


## v5: Training-ready kernels
- **TileLinear backward**: CUDA dX / dW / db using tiled GEMMs.
- **LayerNorm backward**: CUDA dx (dw/db via reductions in Python).
- **FlashAttention backward**: CUDA-assisted path using our batched GEMMs + softmax (no PyTorch matmul).
- **Fused QKV + Bias+GELU**: forward micro-kernel for faster Q path.
- **IO16→compute32**: BF16/FP16 tensors are upcast to FP32 for compute and cast back for outputs. (Next step: true HMMA/TensorCore kernels.)

### Attention controls
- `TESSERA_USE_CUSTOM_KERNELS=1` — prefer CUDA kernels.
- `TESSERA_NAIVE_ATTENTION=1` — force naive attention (our kernels only, no torch.matmul).
- `TESSERA_REFERENCE_KERNELS=1` — use streaming reference kernel for correctness checks.
- `causal=True` (API) — enables triangular mask in fused/naive/reference paths.
- `dropout_mask` — pass a `[B*H, N, N]` mask to fused path when you want dropout applied at probability level.



## v6: Pure fused QKV (single GEMM), FlashAttn dropout/causal, WMMA FP16
- **Fused QKV pack**: `qkv_pack_gemm(X, Wcat, bcat)` computes Q|K|V in one GEMM and applies GELU to Q (optional). Autograd handled at the Python level.
- **FlashAttention forward (ex)**: adds **dropout (in-kernel RNG)** and **causal** support; API: `flash_attn_forward_ex(Q,K,V, dropout_p, causal, seed)` with deterministic RNG.
- **WMMA path**: `tile_linear_wmma` leverages tensor cores when tensors are FP16 and dims are multiples of 16. BF16 currently falls back to FP32 compute via standard path.
- Python ops route TileLinear to WMMA when eligible; otherwise to our FP32 tiled GEMM.

### Examples
```python
# Single-GEMM QKV pack
qkv = fused_qkv_pack(x, Wcat, bcat, gelu_q=True)  # -> [3,B,N,D]

# Fused FlashAttn forward with dropout & causal
out = tessera_flash_attn(q, k, v, dropout_p=0.1, causal=True, seed=12345)
```


## v7: BF16 WMMA, fused QKV backward, HMMA batched GEMM, FlashAttn backward mask+dropout
- **BF16 WMMA**: `tile_linear_wmma_bf16` uses `__nv_bfloat16` with WMMA and accumulates in FP32.
- **Batched GEMM WMMA**: `batched_gemm_wmma(A, B_t)` accelerator for HMMA tiles (FP16/BF16) when shapes are multiples of 16.
- **Fused QKV (single GEMM) autograd**: `_FusedQKVPackFn` runs one GEMM forward; **backward returns dX and dWcat/dbcat** with GELU' folded into Q slice.
- **FlashAttention backward (mask+dropout)**: `flash_attn_backward_ex` supports causal masking and dropout (seeded) to match forward.
- Toggle `TESSERA_USE_QKV_PACK=1` in your attention module to exercise the single-GEMM path end-to-end.

