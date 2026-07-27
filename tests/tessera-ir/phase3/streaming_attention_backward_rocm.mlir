// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt --tessera-tile-ir-lowering='tile-q=17 tile-kv=16 sm=90' \
// RUN:   --rocm-wave-lds-pipeline --rocm-wave-lds-legality \
// RUN:   --lower-tile-to-rocm='arch=gfx1151' --allow-unregistered-dialect %s \
// RUN:   | FileCheck %s --check-prefix=TARGET
// RUN: tessera-opt --tessera-tile-ir-lowering='tile-q=17 tile-kv=16 sm=90' \
// RUN:   --rocm-wave-lds-pipeline --rocm-wave-lds-legality \
// RUN:   --lower-tile-to-rocm='arch=gfx1151' \
// RUN:   --generate-wmma-flash-attn-kernel \
// RUN:   --generate-wmma-flash-attn-bwd-kernel \
// RUN:   --allow-unregistered-dialect %s \
// RUN:   | FileCheck %s --check-prefix=KERNEL

module {
  func.func @attention_forward(
      %q: tensor<1x4x17x64xf16>,
      %key: tensor<1x2x19x64xf16>,
      %v: tensor<1x2x19x64xf16>,
      %bias: tensor<1x4x17x19xf32>
  ) -> tensor<1x4x17x64xf32> {
    %o = "tessera.flash_attn"(%q, %key, %v, %bias)
        <{operandSegmentSizes = array<i32: 1, 1, 1, 1>}> {
      causal = true,
      dropout_p = 0.25 : f64,
      dropout_seed = 37 : i64,
      head_dim = 64 : i64,
      scale = 0.125 : f32,
      softcap = 8.0 : f32,
      tessera.tile_q = 17 : i32,
      tessera.tile_kv = 16 : i32,
      window_left = 64 : i64,
      window_right = 0 : i64
    } : (tensor<1x4x17x64xf16>, tensor<1x2x19x64xf16>,
          tensor<1x2x19x64xf16>, tensor<1x4x17x19xf32>)
          -> tensor<1x4x17x64xf32>
    return %o : tensor<1x4x17x64xf32>
  }

  func.func @attention_backward(
      %do: tensor<1x4x17x64xf16>,
      %q: tensor<1x4x17x64xf16>,
      %key: tensor<1x2x19x64xf16>,
      %v: tensor<1x2x19x64xf16>,
      %bias: tensor<1x4x17x19xf32>
  ) -> (tensor<1x4x17x64xf32>, tensor<1x2x19x64xf32>,
        tensor<1x2x19x64xf32>) {
    %dq, %dk, %dv = "tessera_attn.backward"(
        %do, %q, %key, %v, %bias) {
      causal = true,
      dropout_p = 0.25 : f32,
      dropout_seed = 37 : i64,
      key_block = 16 : i64,
      query_block = 16 : i64,
      scale = 0.125 : f32,
      softcap = 8.0 : f32,
      split_count = 2 : i64,
      window_left = 64 : i64,
      window_right = 0 : i64
    } : (tensor<1x4x17x64xf16>, tensor<1x4x17x64xf16>,
          tensor<1x2x19x64xf16>, tensor<1x2x19x64xf16>,
          tensor<1x4x17x19xf32>)
          -> (tensor<1x4x17x64xf32>, tensor<1x2x19x64xf32>,
              tensor<1x2x19x64xf32>)
    return %dq, %dk, %dv : tensor<1x4x17x64xf32>,
                            tensor<1x2x19x64xf32>,
                            tensor<1x2x19x64xf32>
  }
}

// TARGET: tessera_rocm.flash_attn
// TARGET-SAME: source = "canonical_rank4_kv_scf_for"
// TARGET: tessera_rocm.flash_attn_bwd
// TARGET-SAME: attn_bias = true
// TARGET-SAME: canonical_phase_loops = true
// TARGET-SAME: dropout = true
// TARGET-SAME: gqa = true
// TARGET-SAME: reduction_order = "ascending_split"
// TARGET-SAME: source = "canonical_tensor_backward_scf_for"
// TARGET-SAME: split_count = 2
// TARGET-SAME: split_reduced = true
// TARGET-SAME: workspace_owner = "program_launch"
// TARGET-NOT: tessera_attn.backward_dq_block
// TARGET-NOT: tessera_attn.backward_dkdv_block
// TARGET-NOT: tessera_attn.backward_reduce_split

// KERNEL: gpu.module @attention_forward_mod
// KERNEL: gpu.func @attention_backward_pre(
// KERNEL: gpu.func @attention_backward_dkdv(
// KERNEL: gpu.func @attention_backward_dkdv_reduce(
// KERNEL: gpu.func @attention_backward_dq(
// KERNEL: gpu.func @attention_forward(
// KERNEL-NOT: tessera_rocm.flash_attn_bwd
