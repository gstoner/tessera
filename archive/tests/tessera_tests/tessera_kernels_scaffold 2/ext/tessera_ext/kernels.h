#pragma once
#include <torch/extension.h>

// FP16 WMMA
void gemm_wmma_launcher(at::Tensor A, at::Tensor B, at::Tensor C,
                        float alpha, float beta);

// BF16 WMMA (guarded by TESSERA_ENABLE_BF16)
void gemm_wmma_bf16_launcher(at::Tensor A, at::Tensor B, at::Tensor C,
                             float alpha, float beta);

// Tile reduce
void reduce_tile_sum_launcher(at::Tensor X, at::Tensor Out);

// FlashAttention forward (naive tiled, correctness-first)
void flashattn_naive_fwd_launcher(at::Tensor Q, at::Tensor K, at::Tensor V,
                                  c10::optional<at::Tensor> attn_mask,
                                  double scale,
                                  at::Tensor Out);

// FlashAttention fused backward (naive; with mask/causal/dropout)
void flashattn_bwd_fused_launcher(
    at::Tensor Q, at::Tensor K, at::Tensor V,
    at::Tensor dOut,
    c10::optional<at::Tensor> attn_mask,
    c10::optional<at::Tensor> dropout_mask,
    double scale, double dropout_p, bool is_causal,
    at::Tensor dQ, at::Tensor dK, at::Tensor dV);

// Tiled forward (shared-mem) with causal/mask/dropout
void flashattn_fwd_tiled_launcher(at::Tensor Q, at::Tensor K, at::Tensor V,
                                  c10::optional<at::Tensor> attn_mask,
                                  c10::optional<at::Tensor> dropout_mask,
                                  double scale, double dropout_p, bool is_causal,
                                  at::Tensor Out);
// Tiled fused backward
void flashattn_bwd_tiled_launcher(at::Tensor Q, at::Tensor K, at::Tensor V,
                                  at::Tensor dOut,
                                  c10::optional<at::Tensor> attn_mask,
                                  c10::optional<at::Tensor> dropout_mask,
                                  double scale, double dropout_p, bool is_causal,
                                  at::Tensor dQ, at::Tensor dK, at::Tensor dV);
