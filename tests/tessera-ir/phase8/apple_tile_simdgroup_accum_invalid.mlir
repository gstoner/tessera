// REQUIRES: tessera-apple-backend
// RUN: not tessera-opt %s --split-input-file --tessera-lower-to-apple_gpu-full \
// RUN:   --allow-unregistered-dialect 2>&1 | FileCheck %s
//
// APPLE-ACCUM-1 negative fixture for the TILE-1 value lane
// (tile.matmul -> tessera_apple.gpu.kernel_call "tile_simdgroup_gemm"). The
// lane applies the same accumulator -> result rule as the IR lane
// (MatmulToAppleSimdgroup): an fp16 accumulator returned as a bf16 result
// would round the already-rounded accumulator a second time, so it is refused
// and no kernel_call is emitted. Before this fixture the value lane checked
// only the accumulator and dispatched it.

// CHECK: APPLE_SIMDGROUP_ACCUM_{{UNSUPPORTED}}: apple_gpu TILE-1 simdgroup GEMM (storage 'bf16', accum="fp16"): there is no single-rounding conversion from an f16 accumulator to an bf16 result
// CHECK-NOT: tile_simdgroup_gemm
func.func @bf16_result_from_fp16_accumulator(%a: tensor<8x8xbf16>, %b: tensor<8x8xbf16>) -> tensor<8x8xbf16> {
  %0 = tessera.matmul %a, %b {numeric_policy = {accum = "fp16"}}
      : (tensor<8x8xbf16>, tensor<8x8xbf16>) -> tensor<8x8xbf16>
  return %0 : tensor<8x8xbf16>
}

// -----

// CHECK: APPLE_SIMDGROUP_STORAGE_MISMATCH: apple_gpu TILE-1 simdgroup GEMM: numeric_policy.storage="fp16" does not name the operands' element type bf16
func.func @policy_storage_contradicts_operands(%a: tensor<8x8xbf16>, %b: tensor<8x8xbf16>) -> tensor<8x8xbf16> {
  %0 = tessera.matmul %a, %b {numeric_policy = {storage = "fp16", accum = "fp32"}}
      : (tensor<8x8xbf16>, tensor<8x8xbf16>) -> tensor<8x8xbf16>
  return %0 : tensor<8x8xbf16>
}
