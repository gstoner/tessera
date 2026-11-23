// RUN: tessera-opt --tessera-graph-to-schedule %s | FileCheck %s --check-prefix=SCHED
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile %s | FileCheck %s --check-prefix=TILE
//
// AR-DECODE-PROGRAM-1 S1a: an in-place paged KV write descends Graph ->
// Schedule -> Tile with a content-addressed native contract. The Tile entry
// binds pages read-write (no copy), the position from device memory, and a
// status word for kernel-checked refusals.

module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @paged_write(%pages: tensor<4x16x2x16xf32>, %table: tensor<4xi32>,
                         %values: tensor<1x2x16xf32>, %position: tensor<1xi32>) -> tensor<4x16x2x16xf32>
      attributes {tessera.bindings = ["pages", "page_table", "values", "position"]} {
    %updated = tessera.paged_kv_write %pages, %table, %values, %position
      : (tensor<4x16x2x16xf32>, tensor<4xi32>, tensor<1x2x16xf32>, tensor<1xi32>) -> tensor<4x16x2x16xf32>
    return %updated : tensor<4x16x2x16xf32>
  }
}

// SCHED: tessera.paged_kv_write {{.*}}schedule.artifact_hash = "[[HASH:[0-9a-f]+]]"
// SCHED: schedule.paged_kv_write
// SCHED-SAME: artifact_hash = "[[HASH]]"
// SCHED-SAME: page_ownership = "in_place"
// SCHED-SAME: position_bounds = "kernel_checked_device_position"
// SCHED-SAME: shape = array<i64: 4, 4, 16, 2, 16, 1>

// TILE: llvm.func @tessera_tile_paged_kv_write_f32_direct(
// TILE-SAME: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr
// TILE-SAME: nvvm.kernel
// TILE: tile.paged_kv_write_kernel
// TILE-SAME: page_ownership = "in_place"
// TILE-SAME: route = "direct", storage = "f32", table_storage = "i32"
// TILE-NOT: func.func @paged_write
