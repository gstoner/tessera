// RUN: %tnv --allow-unregistered-dialect --lower-tile-to-nvidia='sm=120' %s | FileCheck %s

// Consumer Blackwell (RTX 50-series, sm_120) is NOT a superset of datacenter
// sm_100: tile.mma must lower to warp-level `mma.sync`, never tcgen05_mma /
// tmem_alloc (those are datacenter sm_100a only). Mirrors the Python guard
// tests/unit/test_target_ir.py::test_lower_tile_to_nvidia_sm120_target_ir_maps_mma_to_warp_level_mma_sync.

module {
  func.func @kernel(%a: f32, %b: f32) {
    %m = "tile.mma"(%a, %b) : (f32, f32) -> f32
    return
  }
}

// CHECK: tessera_nvidia.mma_sync
// CHECK-SAME: arch = "sm_120"
// CHECK-SAME: shape = "m16n8k16"
// CHECK-NOT: tessera_nvidia.tcgen05_mma
// CHECK-NOT: tessera_nvidia.tmem_alloc
