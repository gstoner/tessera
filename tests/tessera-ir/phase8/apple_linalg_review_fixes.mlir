// L-series linalg family — review glass-jaw fixes R1/R2/R3 (2026-06-03).
//
// R1: every linalg tile op is consumed — no orphan tile.<op> remains after
//     lowering, including multi-result ops (svd: 1-in → 3-out).
// R2: the dataflow husk is `tensor.empty` (an explicitly uninitialized value),
//     NOT a rebind to an input operand — so the IR never falsely implies
//     result == input.  (Execution is via the runtime `symbol`; the lowered
//     module is an inspection artifact.)
// R3: the GPU `symbol` is emitted ONLY for ops that actually dispatch on the
//     GPU (status == metal_runtime).  tri_solve is in the runtime envelope;
//     svd has a .mm kernel but is not yet wired into dispatch, so it lowers
//     artifact_only and must NOT advertise a GPU symbol.

// RUN: tessera-opt -tessera-lower-to-apple_cpu-full --allow-unregistered-dialect %s \
// RUN:   | FileCheck %s --check-prefix=CPU
// RUN: tessera-opt -tessera-lower-to-apple_gpu-full --allow-unregistered-dialect %s \
// RUN:   | FileCheck %s --check-prefix=GPU

// Apple-CPU lowering: husk is tensor.empty, no orphan tile ops.
// CPU-LABEL: func.func @svd_multi_result
// CPU: tessera_apple.cpu.vector_op
// CPU-SAME: symbol = "tessera_apple_cpu_svd_f32"
// CPU: tensor.empty
// CPU-NOT: tile.svd
// CPU-NOT: tessera.svd

// Apple-GPU lowering: tri_solve is runtime (symbol present); svd is artifact.
// GPU-LABEL: func.func @tri_solve_runtime
// GPU: tessera_apple.gpu.metal_kernel
// GPU-SAME: status = "metal_runtime"
// GPU-SAME: symbol = "tessera_apple_gpu_tri_solve_f32"

// GPU-LABEL: func.func @svd_multi_result
// GPU: tessera_apple.gpu.metal_kernel
// GPU-SAME: status = "artifact_only"
// GPU-NOT: symbol =
func.func @tri_solve_runtime(%a: tensor<4x4xf32>, %b: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %0 = tessera.tri_solve %a, %b : (tensor<4x4xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

func.func @svd_multi_result(%a: tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<4xf32>, tensor<4x4xf32>) {
  %u, %s, %v = tessera.svd %a : (tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<4xf32>, tensor<4x4xf32>)
  return %u, %s, %v : tensor<6x4xf32>, tensor<4xf32>, tensor<4x4xf32>
}
