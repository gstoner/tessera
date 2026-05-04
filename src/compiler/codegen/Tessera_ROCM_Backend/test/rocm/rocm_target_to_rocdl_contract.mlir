// RUN: %trop --lower-tessera-target-to-rocdl %s | FileCheck %s

module {
  func.func @kernel(%a: f32, %b: f32, %dst: !llvm.ptr, %src: !llvm.ptr, %bytes: i64) {
    %m = tessera_rocm.mfma %a, %b, %a {accum = "f32", arch = "gfx90a", ordinal = 0 : i64, shape = "m16n16k16", source = "tessera.matmul"} : f32, f32, f32 -> f32
    %tok = tessera_rocm.async_copy %dst, %src, %bytes {arch = "gfx90a", dst_space = "lds", src_space = "global"} : !llvm.ptr, !llvm.ptr -> !tessera_rocm.token
    tessera_rocm.wait %tok : !tessera_rocm.token
    return
  }
}

// CHECK: llvm.func @llvm.amdgcn.mfma.contract
// CHECK: llvm.func @llvm.amdgcn.raw.buffer.copy.contract
// CHECK: llvm.func @llvm.amdgcn.s.barrier.contract
// CHECK-NOT: tessera_rocm.mfma
// CHECK-NOT: tessera_rocm.async_copy
// CHECK-NOT: tessera_rocm.wait
