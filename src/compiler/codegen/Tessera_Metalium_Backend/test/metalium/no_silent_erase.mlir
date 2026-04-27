// RUN: tessera-metalium-opt -pass-pipeline='builtin.module(tessera-metalium)' %s 2>&1 | FileCheck %s

module {
  func.func @bad(%x: tensor<4xf32>, %y: tensor<4xf32>) {
    "tessera.tile.copy"(%x, %y) : (tensor<4xf32>, tensor<4xf32>) -> ()
    return
  }
}

// CHECK: failed
