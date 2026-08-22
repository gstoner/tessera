// RUN: %tnv --lower-tessera-nvidia-to-nvvm %s | FileCheck %s
//
// The physical A[4]/B[2]/C[2] register ABI belongs to NVIDIA Target IR; Tile
// uses parameterized !tile.fragment values instead of exposing this vector
// representation directly.

module {
  func.func @fragment_tile(
      %a0: vector<2xf16>, %a1: vector<2xf16>, %a2: vector<2xf16>, %a3: vector<2xf16>,
      %b0: vector<2xf16>, %b1: vector<2xf16>,
      %c0: vector<2xf16>, %c1: vector<2xf16>)
        -> !llvm.struct<(vector<2xf16>, vector<2xf16>)> {
    %d = tessera_nvidia.mma_sync %a0, %a1, %a2, %a3, %b0, %b1, %c0, %c1
        {arch = "sm_120", shape = "m16n8k16", dtype_ab = "f16", dtype_c = "f16"}
        : (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>,
           vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)
          -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
    return %d : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  }
}

// CHECK-LABEL: func.func @fragment_tile
// CHECK: nvvm.mma.sync A[
// CHECK-SAME: shape = #nvvm.shape<m = 16, n = 8, k = 16>
