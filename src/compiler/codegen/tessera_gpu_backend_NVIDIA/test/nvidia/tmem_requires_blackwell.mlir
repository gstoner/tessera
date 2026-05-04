// RUN: not %tnv --allow-unregistered-dialect --lower-tile-to-nvidia='sm=90' %s 2>&1 | FileCheck %s

module {
  func.func @kernel() {
    "tile.tmem.load"() : () -> ()
    return
  }
}

// CHECK: NVIDIA TMEM lowering requires Blackwell SM100+
