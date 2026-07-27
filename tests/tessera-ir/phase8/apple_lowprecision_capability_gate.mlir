// REQUIRES: tessera-apple-backend
//
// Terminal packed storage is a target + operation + physical-consumer decision.
// Apple has no proven physical consumer for sub-byte storage, so the pass must
// leave the policy alone. The NVIDIA run is the *contrast*: without it a green
// Apple run would prove only that the pass did nothing at all.
//
// RUN: tessera-opt --allow-unregistered-dialect \
// RUN:   --tessera-storage-legalize='target=apple_gpu' %s \
// RUN:   | FileCheck %s --check-prefix=APPLE
//
// RUN: tessera-opt --allow-unregistered-dialect \
// RUN:   --tessera-storage-legalize='target=nvidia_sm120' %s \
// RUN:   | FileCheck %s --check-prefix=NVIDIA
//
// A block-scaled cooperative-matrix descriptor is also a capability claim, and
// Apple answers it from its own fragment contract instead of inheriting the
// dtype the shared descriptor permits.
//
// RUN: not tessera-opt --allow-unregistered-dialect \
// RUN:   --pass-pipeline='builtin.module(tessera-apple-threadgroup-pipeline)' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=APPLE-MMA

// APPLE-DTYPE-1-REJECT (2026-07-26). FP8/FP4/MX execution on Apple is gated on
// the macOS-27 Metal tensor toolchain — toolchain, not hardware. Until that
// path runs natively the SDK gate must be *enforced*, not left incidental to
// the absence of an Apple branch in the capability table.

// The Apple lane never stamps packed storage, on any sub-byte format.
// APPLE-NOT: tessera.storage_packed
// APPLE-NOT: tessera.storage_container

// APPLE-MMA: APPLE_MMA_STORAGE_UNSUPPORTED

module {
  llvm.func @nvfp4_decode(
      %src: !llvm.ptr, %scale: !llvm.ptr, %dst: !llvm.ptr,
      %row: i64, %col: i64, %rows: i64, %cols: i64) {
    // The same block-scaled decode on SM120 has a proven physical consumer, so
    // the contrast run does stamp it.
    // NVIDIA: tile.packed_load
    // NVIDIA-SAME: tessera.storage_container = "int8"
    // NVIDIA-SAME: tessera.storage_packed = true
    %tile = tile.packed_load %src, %scale, %row, %col, %rows, %cols {
      numeric_policy = {storage = "nvfp4", accum = "fp32"},
      tile.layout = #tile.layout<shard = [8, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 9>,
      tile.packed_view = #tile.packed_view<
        format = #tile.packed_format<logical = "nvfp4", container = "int8",
          logical_bits = 4, elements_per_container = 2,
          signedness = "format_defined", encoding = "nv_e2m1",
          lane_order = "low_to_high">,
        packing_axis = 1, strides = [9, 1], alignment = 1, offset = 1,
        scale = #tile.scale_layout<dtype = "ue4m3", block_size = 16, axis = 1,
          layout = "row_major", stride = 5, alignment = 1, offset = 2>>
    } : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> !tile.tile
    tile.store %tile, %dst, %row, %col {
      numeric_policy = {storage = "nvfp4", accum = "fp32"},
      tile.layout = #tile.layout<shard = [8, 8] : [8, 1] on ["laneid", "reg"], replica = [] : [] on [], offset = 0>,
      tile.memory = #tile.memory_layout<space = "gmem", order = "row_major", leading_dim = 17>
    } : !tile.tile, !llvm.ptr, i64, i64
    llvm.return
  }

  llvm.func @int4_packed_matmul(
      %a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr,
      %m: i64, %n: i64, %k: i64) {
    // A packed int4 contraction envelope: stamped on SM120, never on Apple,
    // and additionally rejected outright by the Apple capability pass because
    // simdgroup_matrix has no int4 cooperative-matrix route.
    // NVIDIA: tile.matmul_kernel
    // NVIDIA-SAME: tessera.storage_packed = true
    tile.matmul_kernel %a, %b, %d, %m, %n, %k {
      numeric_policy = {storage = "int4", accum = "int32"},
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = 32, a = "int4", b = "int4", acc = "int32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "i32">,
      warps = 1 : i64, staging = "global"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }
}
