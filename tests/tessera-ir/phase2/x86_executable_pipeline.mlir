// RUN: tessera-opt --pass-pipeline='builtin.module(tessera-x86-executable{family=matmul input=tile output=target arch=x86_64_avx512})' %s | FileCheck %s
// RUN: not tessera-opt --pass-pipeline='builtin.module(tessera-x86-executable{family=softmax input=tile output=target arch=x86_64_avx512})' %s 2>&1 | FileCheck %s --check-prefix=MISMATCH

module {
  llvm.func @scheduled_matmul(%a: !llvm.ptr, %b: !llvm.ptr,
                              %o: !llvm.ptr, %m: i64,
                              %n: i64, %k: i64) {
    tile.matmul_kernel %a, %b, %o, %m, %n, %k {
      mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16,
                           a = "f32", b = "f32", acc = "f32",
                           a_layout = "row_major", b_layout = "col_major",
                           k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none",
                                output = "f32">,
      warps = 1 : i64, staging = "global"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }
}

// CHECK: module attributes
// CHECK-SAME: tessera.pipeline.arch = "x86_64_avx512"
// CHECK-SAME: tessera.pipeline.backend_codegen = "prebuilt_native_shared_object"
// CHECK-SAME: tessera.pipeline.family = "matmul"
// CHECK-SAME: tessera.pipeline.target_ir_consumer = "tessera_x86"
// CHECK: tessera_x86.abi_call {symbol = "tessera_x86_avx512_gemm_f32"}
// CHECK: call @tessera_x86_avx512_gemm_f32
// CHECK-NOT: tile.matmul_kernel

// MISMATCH: x86 family plugin 'softmax' cannot consume tile.matmul_kernel
