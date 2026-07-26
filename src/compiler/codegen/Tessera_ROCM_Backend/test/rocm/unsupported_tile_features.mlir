// RUN: not %trop --allow-unregistered-dialect --lower-tile-to-rocm %s 2>&1 | FileCheck %s

module {
  func.func @kernel() {
    %handle = "tile.tmem.allocate"() {bytes = 64 : i64, alignment = 16 : i64}
        : () -> !tile.tmem
    %value = "tile.tmem.load"(%handle) : (!tile.tmem) -> f32
    return
  }
}

// CHECK: ROCm lowering does not support TMEM operations
