// RUN: tessera-opt %s --tessera-graph-to-schedule | FileCheck %s --check-prefix=SCHEDULE
// RUN: tessera-opt %s --tessera-graph-to-schedule --tessera-schedule-to-tile | FileCheck %s --check-prefix=TILE

// SCHEDULE: schedule.moe_dispatch
// SCHEDULE-SAME: artifact_hash
// TILE: llvm.func @tessera_tile_moe_dispatch_f32_direct
// TILE: tile.moe_dispatch_kernel
module attributes {tessera.ir.version = "1.0", tessera.target = "rocm_gfx1151", tessera.arch = "gfx1151"} {
  func.func @gfx1151_moe_dispatch(%x: tensor<7x13xf32>, %token: tensor<9xi32>) -> tensor<9x13xf32> attributes {tessera.bindings = ["x", "token", "o"]} {
    %o = tessera.moe_dispatch %x, %token {tessera.effect_kind = "collective"} : (tensor<7x13xf32>, tensor<9xi32>) -> tensor<9x13xf32>
    return %o : tensor<9x13xf32>
  }
}
