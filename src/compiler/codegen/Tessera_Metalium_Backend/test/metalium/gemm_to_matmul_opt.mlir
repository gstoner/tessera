\
// RUN: tessera-metalium-opt %s -pass-pipeline="tessera-metalium" | FileCheck %s
// REQUIRES: tessera_metalium_opt

module {
  func.func @gemm() {
    "tessera.tile.gemm"() { tile_m = 64, tile_n = 64, tile_k = 32 } : () -> ()
    return
  }
}

// CHECK: tessera_metalium.matmul
// CHECK-SAME: tile = [64, 64, 32]
