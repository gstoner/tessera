// Apple Value Target IR sprint — ODS roundtrip for the value-producing ops
// (cpu.call, gpu.kernel_call, gpu.package_call).  Unlike the attribute-only
// artifact ops, these carry real SSA operands + results.
//
// RUN: tessera-opt %s | FileCheck %s

// CHECK-LABEL: func.func @cpu_call
// CHECK: tessera_apple.cpu.call
// CHECK-SAME: op_kind = "cholesky_solve"
// CHECK-SAME: symbol = "tessera_apple_cpu_cholesky_solve_f32"
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>
func.func @cpu_call(%a: tensor<4x4xf32>, %b: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %0 = tessera_apple.cpu.call %a, %b
      {op_kind = "cholesky_solve", symbol = "tessera_apple_cpu_cholesky_solve_f32",
       abi = "lapack_spotrs", status = "executable", framework = "Accelerate"}
      : (tensor<4x4xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

// CHECK-LABEL: func.func @gpu_kernel_call
// CHECK: tessera_apple.gpu.kernel_call
// CHECK-SAME: status = "executable"
// CHECK-SAME: symbol = "tessera_apple_gpu_cholesky_f32"
func.func @gpu_kernel_call(%a: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = tessera_apple.gpu.kernel_call %a
      {op_kind = "cholesky", symbol = "tessera_apple_gpu_cholesky_f32",
       abi = "msl", status = "executable", framework = "Metal"}
      : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// Multi-result + package lane + optional argument_layout attr.
// CHECK-LABEL: func.func @gpu_package_call
// CHECK: tessera_apple.gpu.package_call
// CHECK-SAME: argument_layout = "row_major"
func.func @gpu_package_call(%a: tensor<6x4xf32>)
    -> (tensor<6x4xf32>, tensor<4xf32>, tensor<4x4xf32>) {
  %u, %s, %v = tessera_apple.gpu.package_call %a
      {op_kind = "svd", symbol = "tessera_apple_gpu_svd_f32",
       status = "executable", framework = "Metal",
       argument_layout = "row_major"}
      : (tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<4xf32>, tensor<4x4xf32>)
  return %u, %s, %v : tensor<6x4xf32>, tensor<4xf32>, tensor<4x4xf32>
}
