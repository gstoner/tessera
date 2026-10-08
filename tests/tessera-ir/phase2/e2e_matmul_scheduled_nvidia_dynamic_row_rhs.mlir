// REQUIRES: tessera-nvidia-backend
// RUN: tessera-opt --tessera-graph-to-schedule %s | FileCheck %s --check-prefix=SCHEDULE
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile %s | FileCheck %s --check-prefix=TILE

module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {
  func.func @bounded_dynamic_row_rhs(
      %a: tensor<?x?xf16>, %b: tensor<?x?xf16>) -> tensor<?x?xf32> {
    %d = tessera.matmul %a, %b {
      shape_bounds = [32, 24, 32], rhs_storage_order = "row_major"}
      : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf32>
    return %d : tensor<?x?xf32>
  }
}
// SCHEDULE: schedule.matmul
// SCHEDULE-SAME: b_layout = "row_major"
// SCHEDULE: shape_key = "M=32;N=24;K=32;dtype=f16"
// TILE-LABEL: llvm.func @bounded_dynamic_row_rhs_row_rhs_kernel
// TILE: scf.for
// TILE: tile.view %arg1
// TILE-SAME: order = "row_major"
// TILE: tile.fragment_pack
// TILE: tile.fragment_pack
// TILE-SAME: transpose
// TILE: tile.mma
// TILE-NOT: tessera.matmul
