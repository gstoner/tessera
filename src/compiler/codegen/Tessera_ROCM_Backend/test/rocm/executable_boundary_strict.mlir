// RUN: not %trop --verify-rocm-executable %s 2>&1 | FileCheck %s

// Nothing target-only or fabricated may cross the binary boundary.
module {
  llvm.func @llvm.amdgcn.placeholder.contract()

  func.func @undef_result() -> i32 {
    %value = llvm.mlir.undef : i32
    return %value : i32
  }

  func.func @target_op(%a: f32, %b: f32) -> f32 {
    %value = tessera_rocm.wmma %a, %b, %a
        {arch = "gfx1151", shape = "m16n16k16"}
        : f32, f32, f32 -> f32
    return %value : f32
  }
}

// CHECK-DAG: contract-marker symbol is forbidden in an executable artifact
// CHECK-DAG: undef is forbidden at the strict ROCm executable boundary
// CHECK-DAG: operation survived the strict ROCm executable boundary
