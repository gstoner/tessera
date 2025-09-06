// RUN: mlir-opt %s | FileCheck %s
// Tile→Target (CPU/AVX2) — smoke sample
module {
  func.func @ebt_cpu_energy_tile_lowered(%a: vector<8xf32>, %b: vector<8xf32>) -> vector<8xf32> {
    %c = vector.fma %a, %b, %a : vector<8xf32>
    return %c : vector<8xf32>
  }
}
// CHECK-LABEL: func @ebt_cpu_energy_tile_lowered
// CHECK: vector.fma
