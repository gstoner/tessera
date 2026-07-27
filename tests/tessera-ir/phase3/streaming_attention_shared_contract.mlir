// Shared forward modifiers and tensor-valued deterministic backward loops.
//
// RUN: tessera-opt --tessera-tile-ir-lowering='tile-q=17 tile-kv=16 sm=90' \
// RUN:   --allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: func.func @forward_bias_softcap
// CHECK: tessera_attn.scaled_dot_product
// CHECK: tessera_attn.score_bias
// CHECK: tessera_attn.softcap
// CHECK-SAME: cap = 3.000000e+00
// CHECK: tessera_attn.boundary_mask
// CHECK: tessera_attn.streaming_update
// CHECK-NOT: tessera.flash_attn
func.func @forward_bias_softcap(
    %q: tensor<1x2x17x64xbf16>,
    %k: tensor<1x1x19x64xbf16>,
    %v: tensor<1x1x19x64xbf16>,
    %bias: tensor<1x2x17x19xf32>) -> tensor<1x2x17x64xf32> {
  %out = "tessera.flash_attn"(%q, %k, %v, %bias)
      <{operandSegmentSizes = array<i32: 1, 1, 1, 1>}> {
    causal = true,
    head_dim = 64 : i64,
    scale = 0.125 : f32,
    softcap = 3.0 : f32,
    tessera.tile_q = 17 : i32,
    tessera.tile_kv = 16 : i32,
    window_left = 8 : i64,
    window_right = 0 : i64
  } : (tensor<1x2x17x64xbf16>, tensor<1x1x19x64xbf16>,
       tensor<1x1x19x64xbf16>, tensor<1x2x17x19xf32>)
       -> tensor<1x2x17x64xf32>
  return %out : tensor<1x2x17x64xf32>
}

// CHECK-LABEL: func.func @backward_split_reduced
// CHECK: tessera_attn.backward_dq_block
// CHECK: tessera.attention_backward_phase = "dq.key_block"
// CHECK: tessera.attention_backward_phase = "dq.query_block"
// CHECK: tessera.attention_backward_phase = "dq.query_head"
// CHECK: tessera.attention_backward_phase = "dq.batch"
// CHECK: tensor<2x1x1x19x64xf32>
// CHECK: tensor<2x1x1x19x64xf32>
// CHECK: tessera_attn.backward_dkdv_block
// CHECK: tessera.workspace_owner = "program_launch"
// CHECK: tessera_attn.backward_reduce_split
// CHECK: tessera.reduction_order = "ascending_split"
// CHECK-NOT: tessera_attn.backward
func.func @backward_split_reduced(
    %do: tensor<1x2x17x64xf32>,
    %q: tensor<1x2x17x64xbf16>,
    %k: tensor<1x1x19x64xbf16>,
    %v: tensor<1x1x19x64xbf16>,
    %bias: tensor<1x17x19xf32>)
    -> (tensor<1x2x17x64xf32>, tensor<1x1x19x64xf32>,
        tensor<1x1x19x64xf32>) {
  %dq, %dk, %dv = "tessera_attn.backward"(%do, %q, %k, %v, %bias) {
    causal = true,
    dropout_p = 2.500000e-01 : f32,
    dropout_seed = 37 : i64,
    key_block = 16 : i64,
    query_block = 16 : i64,
    scale = 1.250000e-01 : f32,
    softcap = 3.000000e+00 : f32,
    split_count = 2 : i64,
    window_left = 8 : i64,
    window_right = 0 : i64
  } : (tensor<1x2x17x64xf32>, tensor<1x2x17x64xbf16>,
       tensor<1x1x19x64xbf16>, tensor<1x1x19x64xbf16>,
       tensor<1x17x19xf32>)
       -> (tensor<1x2x17x64xf32>, tensor<1x1x19x64xf32>,
           tensor<1x1x19x64xf32>)
  return %dq, %dk, %dv : tensor<1x2x17x64xf32>,
                           tensor<1x1x19x64xf32>,
                           tensor<1x1x19x64xf32>
}
