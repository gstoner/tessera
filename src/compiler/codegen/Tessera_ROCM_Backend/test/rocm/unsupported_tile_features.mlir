// RUN: not %trop --allow-unregistered-dialect --lower-tile-to-rocm %s 2>&1 | FileCheck %s

module {
  func.func @kernel() {
    "tile.tmem.load"() : () -> ()
    return
  }
}

// CHECK: ROCm lowering does not support TMEM operations
