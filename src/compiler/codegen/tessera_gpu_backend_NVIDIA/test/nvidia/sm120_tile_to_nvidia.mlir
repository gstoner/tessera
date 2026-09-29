// RUN: not %tnv --allow-unregistered-dialect --lower-tile-to-nvidia='sm=120' %s 2>&1 | FileCheck %s

// A tensor-valued tile.mma has no physical lane fragment mapping. The
// positive typed accumulator-loop fixture proves sm_120 mma.sync lowering.

module {
  func.func @kernel(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>) {
    %m = "tile.mma"(%a, %b) : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    return
  }
}

// CHECK: sm_120 tile.mma requires typed fragment registers and an accumulator
