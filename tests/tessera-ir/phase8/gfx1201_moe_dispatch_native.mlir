// RUN: tessera-opt %s --tessera-graph-to-schedule | FileCheck %s --check-prefix=SCHEDULE
// RUN: tessera-opt %s --tessera-graph-to-schedule --tessera-schedule-to-tile | FileCheck %s --check-prefix=TILE

// gfx1201 joins the MoE token-gather owner on Tajasarus device proof
// (sync E2E-REAL-6-rocm-moe-gfx1201-2026-11-04); the contract names gfx1201.
// SCHEDULE: schedule.moe_dispatch
// SCHEDULE-SAME: artifact_hash
// SCHEDULE-SAME: arch = "gfx1201"
// SCHEDULE-SAME: target = "rocm_gfx1201"
// TILE: llvm.func @tessera_tile_moe_dispatch_f32_direct
// TILE-SAME: arch = "gfx1201"
// TILE: tile.moe_dispatch_kernel
module attributes {tessera.ir.version = "1.0", tessera.target = "rocm_gfx1201", tessera.arch = "gfx1201"} {
  func.func @gfx1201_moe_dispatch(%x: tensor<7x13xf32>, %token: tensor<9xi32>) -> tensor<9x13xf32> attributes {tessera.bindings = ["x", "token", "o"]} {
    %o = tessera.moe_dispatch %x, %token {tessera.effect_kind = "collective"} : (tensor<7x13xf32>, tensor<9xi32>) -> tensor<9x13xf32>
    return %o : tensor<9x13xf32>
  }
}
