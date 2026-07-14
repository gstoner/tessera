// RUN: %tnv --allow-unregistered-dialect --lower-tile-to-nvidia='sm=120' --lower-tessera-nvidia-to-nvvm %s | FileCheck %s
//
// Canonical fragment-form Tile MMA feeds the real NVVM intrinsic. Scalar Tile
// MMA remains the conservative marker path; this fixture proves the explicit
// A[4]/B[2]/C[2] handoff does not need a second target-IR representation.

module {
  func.func @fragment_tile(
      %a0: vector<2xf16>, %a1: vector<2xf16>, %a2: vector<2xf16>, %a3: vector<2xf16>,
      %b0: vector<2xf16>, %b1: vector<2xf16>,
      %c0: vector<2xf16>, %c1: vector<2xf16>)
        -> !llvm.struct<(vector<2xf16>, vector<2xf16>)> {
    %d = "tile.mma"(%a0, %a1, %a2, %a3, %b0, %b1, %c0, %c1)
        : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>,
           vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)
          -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
    return %d : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  }
}

// CHECK-LABEL: func.func @fragment_tile
// CHECK: nvvm.mma.sync A[
// CHECK-SAME: shape = #nvvm.shape<m = 16, n = 8, k = 16>
