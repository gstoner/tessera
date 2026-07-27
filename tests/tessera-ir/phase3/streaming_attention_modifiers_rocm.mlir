// The direct shared recurrence carries per-head bias and softcap into the
// gfx1151 physical adapter; the adapter selects AMD scheduling without
// reconstructing score semantics.
//
// RUN: tessera-opt --tessera-tile-ir-lowering='tile-q=17 tile-kv=16 sm=90' \
// RUN:   --rocm-wave-lds-pipeline --rocm-wave-lds-legality \
// RUN:   --lower-tile-to-rocm='arch=gfx1151' --allow-unregistered-dialect %s \
// RUN:   | FileCheck %s --check-prefix=TARGET
// RUN: tessera-opt --tessera-tile-ir-lowering='tile-q=17 tile-kv=16 sm=90' \
// RUN:   --rocm-wave-lds-pipeline --rocm-wave-lds-legality \
// RUN:   --lower-tile-to-rocm='arch=gfx1151' \
// RUN:   --generate-wmma-flash-attn-kernel --allow-unregistered-dialect %s \
// RUN:   | FileCheck %s --check-prefix=KERNEL

// TARGET: tessera_rocm.flash_attn
// TARGET-SAME: attn_bias = true
// TARGET-SAME: canonical_kv_loop = true
// TARGET-SAME: logit_softcap = true
// TARGET-SAME: source = "canonical_rank4_kv_scf_for"
// TARGET-NOT: tessera_attn.score_bias
// TARGET-NOT: tessera_attn.softcap

// KERNEL: gpu.module @forward_bias_softcap_mod
// KERNEL: gpu.func @forward_bias_softcap
// KERNEL-SAME: memref<?xf32>
// KERNEL-NOT: tessera_rocm.flash_attn

module {
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
}
