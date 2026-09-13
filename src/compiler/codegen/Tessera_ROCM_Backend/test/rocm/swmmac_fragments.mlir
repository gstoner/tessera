// RUN: %trop --lower-tessera-target-to-rocdl %s | FileCheck %s
// Internal physical fragment ABI, not public dense/sparse Graph admission.
module {
  func.func @half(%a: vector<8xf16>, %b: vector<16xf16>, %c: vector<8xf32>, %idx: i32) -> vector<8xf32> {
    %r = tessera_rocm.swmmac %a, %b, %c, %idx {arch = "gfx1201"} : vector<8xf16>, vector<16xf16>, vector<8xf32> -> vector<8xf32>
    return %r : vector<8xf32>
  }
  func.func @bfloat(%a: vector<8xbf16>, %b: vector<16xbf16>, %c: vector<8xf32>, %idx: i32) -> vector<8xf32> {
    %r = tessera_rocm.swmmac %a, %b, %c, %idx {arch = "gfx1201"} : vector<8xbf16>, vector<16xbf16>, vector<8xf32> -> vector<8xf32>
    return %r : vector<8xf32>
  }
}
// CHECK-LABEL: func.func @half
// CHECK: llvm.call_intrinsic "llvm.amdgcn.swmmac.f32.16x16x32.f16"
// CHECK-LABEL: func.func @bfloat
// CHECK: llvm.bitcast
// CHECK: llvm.call_intrinsic "llvm.amdgcn.swmmac.f32.16x16x32.bf16"
