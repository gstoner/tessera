// The bias-free route still carries the shared mandatory bias operand, but it
// must be a canonical zero tensor rather than an invented launch argument.
//
// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt --tessera-tile-ir-lowering='tile-q=17 tile-kv=16 sm=90' \
// RUN:   --rocm-wave-lds-pipeline --rocm-wave-lds-legality \
// RUN:   --lower-tile-to-rocm='arch=gfx1151' --allow-unregistered-dialect %s \
// RUN:   | FileCheck %s

func.func @attention_backward_nobias(
    %do: tensor<1x4x17x64xbf16>,
    %q: tensor<1x4x17x64xbf16>,
    %key: tensor<1x2x19x64xbf16>,
    %v: tensor<1x2x19x64xbf16>
) -> (tensor<1x4x17x64xf32>, tensor<1x2x19x64xf32>,
      tensor<1x2x19x64xf32>) {
  %zero_bias = arith.constant dense<0.000000e+00> : tensor<1x17x19xf32>
  %dq, %dk, %dv = "tessera_attn.backward"(
      %do, %q, %key, %v, %zero_bias) {
    causal = false,
    dropout_p = 0.000000e+00 : f32,
    dropout_seed = 0 : i64,
    key_block = 16 : i64,
    query_block = 16 : i64,
    scale = 1.250000e-01 : f32,
    softcap = 0.000000e+00 : f32,
    split_count = 2 : i64,
    window_left = -1 : i64,
    window_right = -1 : i64
  } : (tensor<1x4x17x64xbf16>, tensor<1x4x17x64xbf16>,
        tensor<1x2x19x64xbf16>, tensor<1x2x19x64xbf16>,
        tensor<1x17x19xf32>)
        -> (tensor<1x4x17x64xf32>, tensor<1x2x19x64xf32>,
            tensor<1x2x19x64xf32>)
  return %dq, %dk, %dv : tensor<1x4x17x64xf32>,
                          tensor<1x2x19x64xf32>,
                          tensor<1x2x19x64xf32>
}

// CHECK: tessera_rocm.flash_attn_bwd
// CHECK-SAME: attn_bias = false
// CHECK-SAME: canonical_phase_loops = true
// CHECK-SAME: dropout = false
// CHECK-SAME: dtype = "bf16"
// CHECK-SAME: logit_softcap = false
// CHECK-SAME: source = "canonical_tensor_backward_scf_for"
// CHECK-SAME: split_reduced = true
// CHECK-NOT: func.func
