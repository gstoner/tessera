// REQUIRES: tessera-apple-backend
//
// APPLE-MATMUL2D-1 admission (2026-09-15). The default `tessera-lower-to-apple_gpu`
// pipeline admits the Metal 4 matmul2d family for the 8/4-bit storage pairs
// ONLY: the paired corpus (M1 Max, benchmarks/baselines/apple_matmul2d_route_
// corpus_20260915) retained the incumbent for every f16 shape and found the
// compiled bf16 route within +-10% of the runtime's own bf16 entry, while the
// low-precision pairs had no executable route at all -- this pipeline used to
// emit an MPSGraph `matmul_contract` claim for an FP8 GEMM that MPSGraph
// cannot run. f16/bf16 keep the APPLE-TILE-2 incumbent.
//
// RUN: tessera-opt %s --tessera-tiling --tessera-lower-to-apple_gpu \
// RUN:   --allow-unregistered-dialect | FileCheck %s --check-prefix=PIPE
// RUN: tessera-opt %s --tessera-tiling --tessera-apple-canonical-gemm-matmul2d=admit=lowp \
// RUN:   --allow-unregistered-dialect | FileCheck %s --check-prefix=LOWP
// RUN: not tessera-opt %s --tessera-tiling --tessera-apple-canonical-gemm-matmul2d=admit=bogus \
// RUN:   --allow-unregistered-dialect 2>&1 | FileCheck %s --check-prefix=BAD

// f16 stays on the incumbent simdgroup route.
// PIPE-LABEL: func.func @gemm_f16
// PIPE-NOT: mtl4_matmul2d
// PIPE: tessera_apple.gpu.kernel_call %arg0, %arg1
// PIPE-SAME: op_kind = "tile_simdgroup_gemm"
// LOWP-LABEL: func.func @gemm_f16
// LOWP: scf.for
// LOWP-NOT: tessera_apple.gpu.matmul2d
func.func @gemm_f16(%a: tensor<64x128xf16>, %b: tensor<128x96xf16>) -> tensor<64x96xf32> {
  %0 = "tessera.matmul"(%a, %b) : (tensor<64x128xf16>, tensor<128x96xf16>) -> tensor<64x96xf32>
  return %0 : tensor<64x96xf32>
}

// bf16 stays on the incumbent too (within +-10% of the runtime's bf16 entry).
// PIPE-LABEL: func.func @gemm_bf16
// PIPE-NOT: mtl4_matmul2d
// PIPE: op_kind = "tile_simdgroup_gemm"
func.func @gemm_bf16(%a: tensor<64x128xbf16>, %b: tensor<128x96xbf16>) -> tensor<64x96xf32> {
  %0 = "tessera.matmul"(%a, %b) : (tensor<64x128xbf16>, tensor<128x96xbf16>) -> tensor<64x96xf32>
  return %0 : tensor<64x96xf32>
}

// An FP8 GEMM is now an executable value call instead of an MPSGraph claim.
// PIPE-LABEL: func.func @gemm_e4m3
// PIPE-NOT: matmul_contract
// PIPE: tessera_apple.gpu.kernel_call %arg0, %arg1
// PIPE-SAME: op_kind = "mtl4_matmul2d"
// PIPE-SAME: status = "executable"
// PIPE-SAME: symbol = "tessera_apple_gpu_mtl4_matmul2d_view"
// PIPE-SAME: tessera_apple.pair = 0
// PIPE-NOT: matmul_contract
// LOWP-LABEL: func.func @gemm_e4m3
// LOWP-NOT: scf.for
// LOWP: tessera_apple.gpu.matmul2d
func.func @gemm_e4m3(%a: tensor<128x256xf8E4M3FN>, %b: tensor<256x128xf8E4M3FN>) -> tensor<128x128xf32> {
  %0 = "tessera.matmul"(%a, %b) : (tensor<128x256xf8E4M3FN>, tensor<256x128xf8E4M3FN>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// The weight-only pair (half activations, FP4 weights) is low precision too.
// PIPE-LABEL: func.func @gemm_half_e2m1
// PIPE: op_kind = "mtl4_matmul2d"
// PIPE-SAME: tessera_apple.pair = 5
func.func @gemm_half_e2m1(%a: tensor<64x128xf16>, %b: tensor<128x256xf4E2M1FN>) -> tensor<64x256xf32> {
  %0 = "tessera.matmul"(%a, %b) : (tensor<64x128xf16>, tensor<128x256xf4E2M1FN>) -> tensor<64x256xf32>
  return %0 : tensor<64x256xf32>
}

// The admission set is closed.
// BAD: APPLE_MATMUL2D_ADMIT: admit must be 'all' or 'lowp', got 'bogus'
