// L-series linalg pilot — L7: full Graph→Schedule→Tile→Target Apple spine.
//
// A single pipeline alias drives the entire stack for cholesky:
//   tessera.cholesky (Graph IR)
//     → effect-annotation → distribution-lowering (Schedule)
//     → tiling (tile.cholesky, Tile IR)
//     → tile-to-apple_{cpu,gpu} (tessera_apple.* Target IR + runtime symbol)
//
// This is the reusable template: every other linalg op (tri_solve, svd) follows
// the same rails.  The emitted Target IR names the C ABI `symbol` the runtime
// executes (proven numerically by tests/unit/test_apple_cholesky_seam_closure.py).
//
// RUN: tessera-opt -tessera-lower-to-apple_cpu-full --allow-unregistered-dialect %s \
// RUN:   | FileCheck %s --check-prefix=CPU
// RUN: tessera-opt -tessera-lower-to-apple_gpu-full --allow-unregistered-dialect %s \
// RUN:   | FileCheck %s --check-prefix=GPU

// CPU-LABEL: func.func @chol_e2e
// CPU: tessera_apple.cpu.vector_op
// CPU-SAME: abi = "lapack_spotrf"
// CPU-SAME: symbol = "tessera_apple_cpu_cholesky_f32"
// CPU-NOT: = tessera.cholesky %

// GPU-LABEL: func.func @chol_e2e
// GPU: tessera_apple.gpu.metal_kernel
// GPU-SAME: status = "metal_runtime"
// GPU-SAME: symbol = "tessera_apple_gpu_cholesky_f32"
// GPU-NOT: = tessera.cholesky %
func.func @chol_e2e(%a: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = tessera.cholesky %a : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
