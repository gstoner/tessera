// REQUIRES: tessera-apple-backend
//
// The input is a plain Graph-IR matmul: the *shared* TilingPass turns it into
// the canonical M/N/K contract, and only then does Apple claim it. Driving both
// passes in one run is the point — it proves Apple consumes the real shared
// form rather than a hand-written imitation of it.
//
// RUN: tessera-opt %s --tessera-tiling --tessera-apple-canonical-gemm \
// RUN:   --allow-unregistered-dialect | FileCheck %s
//
// The dtype/shape/accumulator rejections are owned by
// tests/unit/test_apple_canonical_gemm.py, which also carries the exact-device
// execute-and-compare rows.

// APPLE-TILE-2 (2026-07-26). `CORE-GEMM-KLOOP-2026-07-25` states a dense
// contraction as three nested `scf.for` loops carrying an FP32 accumulator and
// staged `!tile.pipeline_state`. Apple's physical answer is a single
// `simdgroup_matrix` dispatch: the loop is a semantic contract, not a schedule
// Metal must reproduce statement by statement.
//
// Recognition is not promotion. Value-mode Accelerate/MPS remains the incumbent
// production route until a two-run paired corpus says otherwise.

// The nest is consumed, not left beside the dispatch.
// CHECK-LABEL: func.func @gemm
// CHECK-NOT: scf.for
// CHECK: tessera_apple.gpu.kernel_call
// CHECK-SAME: op_kind = "tile_simdgroup_gemm"
// CHECK-SAME: symbol = "tessera_apple_gpu_tile_simdgroup_gemm_f16"
// The dispatch names the shared contract it answers, and carries the loop's
// own tile decision plus the ragged-tail guarantee it must honor.
// CHECK-SAME: tessera_apple.accumulate = "fp32"
// CHECK-SAME: tessera_apple.canonical_k_loop = true
// CHECK-SAME: tessera_apple.ragged_zero_pad = true
// CHECK-SAME: tessera_apple.tile_k = 16
// CHECK-SAME: tessera_apple.tile_m = 16
// CHECK-SAME: tessera_apple.tile_n = 16
// CHECK-NOT: scf.for
func.func @gemm(%a: tensor<64x32xf16>, %b: tensor<32x48xf16>) -> tensor<64x48xf32> {
  %0 = "tessera.matmul"(%a, %b)
      : (tensor<64x32xf16>, tensor<32x48xf16>) -> tensor<64x48xf32>
  return %0 : tensor<64x48xf32>
}

// bf16 selects its own storage-matched symbol rather than converting inputs.
// CHECK-LABEL: func.func @gemm_bf16
// CHECK: tessera_apple.gpu.kernel_call
// CHECK-SAME: dtype = "bf16"
// CHECK-SAME: symbol = "tessera_apple_gpu_tile_simdgroup_gemm_bf16"
func.func @gemm_bf16(%a: tensor<32x16xbf16>, %b: tensor<16x32xbf16>)
    -> tensor<32x32xf32> {
  %0 = "tessera.matmul"(%a, %b)
      : (tensor<32x16xbf16>, tensor<16x32xbf16>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}
