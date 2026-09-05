// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --split-input-file %s | FileCheck %s
module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @norm(%x: tensor<3x17xbf16>) -> tensor<3x17xbf16> {
    %0 = "tessera.rmsnorm"(%x) {eps = 1.0e-5 : f64} : (tensor<3x17xbf16>) -> tensor<3x17xbf16>
    return %0 : tensor<3x17xbf16>
  }
}
// CHECK-NOT: func.func
// CHECK-LABEL: llvm.func @tessera_tile_norm_rmsnorm_bf16_
// CHECK-SAME: nvvm.kernel
// CHECK: arith.constant
// CHECK: tile.norm_kernel
// CHECK-SAME: affine = false
// CHECK-SAME: tessera.norm_epsilon
// CHECK-SAME: tessera.schedule_hash
// CHECK: llvm.return
// -----
module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @attn(%q: tensor<1x2x8x16xf16>, %k: tensor<1x1x8x16xf16>,
                  %v: tensor<1x1x8x16xf16>) -> tensor<1x2x8x16xf32> {
    %0 = tessera.flash_attn %q, %k, %v {
      scale = 0.25, causal = true, window_left = -1, window_right = -1,
      softcap = 0.0, dropout_p = 0.0, dropout_seed = 0, head_dim = 16,
      operandSegmentSizes = array<i32: 1, 1, 1, 0>
    } : (tensor<1x2x8x16xf16>, tensor<1x1x8x16xf16>, tensor<1x1x8x16xf16>) -> tensor<1x2x8x16xf32>
    return %0 : tensor<1x2x8x16xf32>
  }
}
// CHECK-NOT: func.func
// CHECK-LABEL: llvm.func @tessera_tile_attention_f16_causal_
// CHECK-SAME: nvvm.kernel
// CHECK: tile.attention_kernel
// CHECK-SAME: lse_checkpoint = "recompute"
// CHECK-SAME: tessera.backward_lse_policy = "sm120_recompute"
// CHECK-SAME: tessera.schedule_hash
// CHECK-SAME: tessera.workgroup_size = 128
// CHECK: llvm.return
