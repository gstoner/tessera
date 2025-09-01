<!-- MERGE_START:DINOv3_Tessera_Kernels -->
# Kernel Map: DINOv3 ↔ Tessera Tiles (v2)

| Component | Tessera Kernel | Schedule Fields Used | Behavior |
|---|---|---|---|
| QKV / Proj / MLP | `tile_linear` | `block_m`, `block_k` | Blocked GEMM (B,N,K)·(K,M) with optional fused GELU |
| LayerNorm | `layer_norm` | `block_k` | Chunked reductions, fused affine |
| FlashAttention | `flash_attn` | `block_n`, `block_k`, `stages` | Streaming log-sum-exp over K/V blocks |

> The **reference** kernels here are correctness-first. Replace them with your
> optimized Tessera implementations behind the same API to get speedups.
<!-- MERGE_END:DINOv3_Tessera_Kernels -->
