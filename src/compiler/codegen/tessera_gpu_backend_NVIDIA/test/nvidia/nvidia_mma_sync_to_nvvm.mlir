// RUN: %tnv --lower-tessera-nvidia-to-nvvm %s | FileCheck %s
//
// Increment 2 of the Tile IR / native Target IR tail: a fragment-typed
// tessera_nvidia.mma_sync (m16n8k16 f16 contract) lowers to a REAL nvvm.mma.sync
// intrinsic op, not a void marker. The emitted op is validated by the NVVM
// verifier (A=4 / B=2 / C=2 vector<2xf16> fragments, struct result) — a real
// structural correctness signal without a device. The abstract (scalar) form
// carries no fragments and falls through to the honest void marker (Decision
// #21). Parses without --allow-unregistered-dialect (the ops are registered).

module {
  func.func @mma_f16(
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
  // CHECK-LABEL: func.func @mma_f16
  // CHECK: nvvm.mma.sync A[
  // CHECK-SAME: shape = #nvvm.shape<m = 16, n = 8, k = 16>
  // CHECK-SAME: -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>

  // Abstract (scalar) mma_sync carries no fragment contract -> honest marker,
  // never a malformed real intrinsic.
  func.func @mma_abstract(%a: f32, %b: f32) {
    %0 = tessera_nvidia.mma_sync %a, %b
        {arch = "sm_120", shape = "m16n8k16", dtype_ab = "bf16"} : (f32, f32) -> f32
    return
  }
  // CHECK-LABEL: func.func @mma_abstract
  // CHECK: llvm.call @llvm.nvvm.mma.sync.contract
}
