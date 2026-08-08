// RUN: %trop %s | FileCheck %s --check-prefix=TARGET
// RUN: not %trop --lower-tessera-target-to-rocdl %s 2>&1 | FileCheck %s --check-prefix=STRICT

module {
  func.func @kernel(%a: f32, %b: f32, %dst: !llvm.ptr, %src: !llvm.ptr, %bytes: i64) {
    %m = tessera_rocm.mfma %a, %b, %a {accum = "f32", arch = "gfx90a", ordinal = 0 : i64, shape = "m16n16k16", source = "tessera.matmul"} : f32, f32, f32 -> f32
    %tok = tessera_rocm.async_copy %dst, %src, %bytes {arch = "gfx90a", dst_space = "lds", src_space = "global"} : !llvm.ptr, !llvm.ptr -> !tessera_rocm.token
    tessera_rocm.wait %tok : !tessera_rocm.token
    return
  }
}

// TARGET: tessera_rocm.mfma
// TARGET: tessera_rocm.async_copy
// TARGET: tessera_rocm.wait
// TARGET-NOT: .contract
// STRICT: executable ROCm async operations must pass through lower-rocm-async-copy
