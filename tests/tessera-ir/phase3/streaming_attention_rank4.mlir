// Canonical rank-4 attention distribution.
//
// Batch and query-head distribution are explicit scf.for loops. GQA maps each
// query head to its owning KV head before rank-reducing Q/K/V to the one
// canonical rank-2 KV-block recurrence.
//
// RUN: tessera-opt --tessera-tile-ir-lowering='tile-q=17 tile-kv=16 sm=90' \
// RUN:   --allow-unregistered-dialect %s | FileCheck %s
// RUN: tessera-opt --tessera-tile-ir-lowering='tile-q=17 tile-kv=16 sm=90' \
// RUN:   --rocm-wave-lds-pipeline --rocm-wave-lds-legality \
// RUN:   --lower-tile-to-rocm='arch=gfx1151' --allow-unregistered-dialect %s \
// RUN:   | FileCheck %s --check-prefix=ROCM
// RUN: tessera-opt --tessera-tile-ir-lowering='tile-q=17 tile-kv=16 sm=90' \
// RUN:   --rocm-wave-lds-pipeline --rocm-wave-lds-legality \
// RUN:   --lower-tile-to-rocm='arch=gfx1151' \
// RUN:   --generate-wmma-flash-attn-kernel --allow-unregistered-dialect %s \
// RUN:   | FileCheck %s --check-prefix=KERNEL

// CHECK-LABEL: func.func @rank4_gqa
// CHECK: scf.for
// CHECK-SAME: iter_args
// CHECK: scf.for
// CHECK-SAME: iter_args
// CHECK: arith.divui
// CHECK: tensor.extract_slice
// CHECK: tensor.extract_slice
// CHECK: tensor.extract_slice
// CHECK: scf.for
// CHECK-SAME: iter_args
// CHECK: tessera_attn.scaled_dot_product
// CHECK-SAME: scale = 1.250000e-01
// CHECK: tessera_attn.boundary_mask
// CHECK: tessera_attn.streaming_update
// CHECK: tessera.streaming_attention
// CHECK: tensor.insert_slice
// CHECK: tessera.attention_distribution = "query_head"
// CHECK-SAME: tessera.gqa_group_size = 2
// CHECK: tessera.attention_distribution = "batch"
// CHECK-NOT: tessera.flash_attn

// ROCM: tessera_rocm.flash_attn
// ROCM-SAME: arch = "gfx1151"
// ROCM-SAME: canonical_kv_loop = true
// ROCM-SAME: gqa = true
// ROCM-SAME: kv_block = 16
// ROCM-SAME: rank4_distribution = true
// ROCM-SAME: schedule = "gfx1151_wmma_streaming_attention"
// ROCM-SAME: source = "canonical_rank4_kv_scf_for"
// ROCM-SAME: ssa_pipeline = true
// ROCM-NOT: tessera_attn.streaming_update
// ROCM-NOT: scf.for

// KERNEL: gpu.module @rank4_gqa_mod
// KERNEL: gpu.func @rank4_gqa
// KERNEL-SAME: memref<?xbf16>
// KERNEL-SAME: memref<?xf32>
// KERNEL: arith.divui
// KERNEL: scf.for
// KERNEL: tessera_rocm.wmma
// KERNEL-NOT: tessera_rocm.flash_attn
// KERNEL-NOT: tessera_attn.streaming_update

module attributes {
  tessera.ir.version = "1.0",
  tessera.target = {sm = 90 : i32, warps = 4 : i32,
                    smem = 233472 : i64, pipeline_stages = 2 : i32}
} {
  func.func @rank4_gqa(
      %Q: tensor<2x4x17x64xbf16>,
      %K: tensor<2x2x19x64xbf16>,
      %V: tensor<2x2x19x64xbf16>
  ) -> tensor<2x4x17x64xf32> {
    %out = "tessera.flash_attn"(%Q, %K, %V)
        <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> {
      causal = true,
      head_dim = 64 : i64,
      scale = 0.125 : f32,
      tessera.tile_q = 17 : i32,
      tessera.tile_kv = 16 : i32,
      window_left = 8 : i64,
      window_right = 0 : i64
    } : (tensor<2x4x17x64xbf16>, tensor<2x2x19x64xbf16>,
         tensor<2x2x19x64xbf16>) -> tensor<2x4x17x64xf32>
    return %out : tensor<2x4x17x64xf32>
  }
}
