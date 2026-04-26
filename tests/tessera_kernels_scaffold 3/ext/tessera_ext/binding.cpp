#include <torch/extension.h>
#include "kernels.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_wmma", &gemm_wmma_launcher, "WMMA GEMM FP16 (A@B->C)");
  m.def("gemm_wmma_bf16", &gemm_wmma_bf16_launcher, "WMMA GEMM BF16 (A@B->C) [guarded]");
  m.def("reduce_tile_sum", &reduce_tile_sum_launcher, "TileReduce sum");
  m.def("flashattn_naive_fwd", &flashattn_naive_fwd_launcher, "Naive tiled FlashAttention forward");
  m.def("flashattn_bwd_fused", &flashattn_bwd_fused_launcher, "Naive fused FlashAttention backward");
}
