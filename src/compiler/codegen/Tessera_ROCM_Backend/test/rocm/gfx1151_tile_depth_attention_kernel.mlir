// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1151})' %s | FileCheck %s --check-prefix=TARGET
// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1151},generate-rocm-depth-attention-kernel)' %s | FileCheck %s --check-prefix=GENERATED
// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(tessera-rocm-executable{family=depth_attention input=tile output=target arch=gfx1151})' %s | FileCheck %s --check-prefix=PIPELINE-TARGET

module {
  llvm.func @block_attnres_depth_attention(
      %query: !llvm.ptr, %sources: !llvm.ptr, %output: !llvm.ptr) {
    %source_count = llvm.mlir.constant(7 : i64) : i64
    %rows = llvm.mlir.constant(3 : i64) : i64
    %width = llvm.mlir.constant(8 : i64) : i64
    tile.depth_attention_kernel %query, %sources, %output,
        %source_count, %rows, %width {
      accum = "f32", arch = "gfx1151", eps = 1.000000e-06 : f64,
      merge_recurrence = "max_shifted_pairwise_merge_v1",
      reassociable = true, softmax = "f32", source_tile = 7 : i64,
      statistics_recurrence = "rms_key_online_softmax_stats_v1",
      storage = "f32",
      tessera.schedule_hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      tessera.workgroup_size = 256 : i64
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }
}

// TARGET-NOT: tile.depth_attention_kernel
// TARGET: tessera_rocm.depth_attention
// TARGET-SAME: arch = "gfx1151"
// TARGET-SAME: merge_recurrence = "max_shifted_pairwise_merge_v1"
// TARGET-SAME: source_count = 7 : i64
// TARGET-SAME: statistics_recurrence = "rms_key_online_softmax_stats_v1"

// GENERATED-NOT: tile.depth_attention_kernel
// GENERATED-NOT: tessera_rocm.depth_attention
// GENERATED: gpu.module @block_attnres_depth_attention_mod
// GENERATED: gpu.func @block_attnres_depth_attention
// GENERATED-SAME: memref<?xf32>
// GENERATED-SAME: tessera.schedule_hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
// GENERATED: gpu.block_id x
// GENERATED: gpu.thread_id x
// GENERATED: math.sqrt
// GENERATED: gpu.barrier
// GENERATED: math.exp
// GENERATED: arith.maximumf
// GENERATED: arith.divf

// PIPELINE-TARGET: tessera.pipeline.output = "target"
// PIPELINE-TARGET: tessera_rocm.depth_attention
// PIPELINE-TARGET-NOT: {{^ *tile\.depth_attention_kernel}}
// PIPELINE-TARGET-NOT: gpu.module
