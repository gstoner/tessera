// L-series linalg pilot — L7: full Graph→Schedule→Tile→Target Apple spine.
//
// A single pipeline alias drives the entire stack for cholesky:
//   tessera.cholesky (Graph IR)
//     → effect-annotation → distribution-lowering (Schedule)
//     → tiling (tile.cholesky, Tile IR)
//     → tile-to-apple_{cpu,gpu} (tessera_apple.* Target IR + runtime symbol)
//
// This is the reusable template: every other linalg op (tri_solve, svd) follows
// the same rails.  The `-full` pipeline is VALUE-preserving: it emits
// value-producing tessera_apple.{cpu.call,gpu.kernel_call} ops that carry the
// real SSA operands + results (a true semantics-preserving hand-off — the
// return consumes the produced value), name the C ABI `symbol` the runtime
// executes, and leave NO ub.poison / tensor.empty / surviving tile.* in the
// module.  (Numerically proven by test_apple_cholesky_seam_closure.py.)
//
// RUN: tessera-opt -tessera-lower-to-apple_cpu-full %s \
// RUN:   | FileCheck %s --check-prefix=CPU
// RUN: tessera-opt -tessera-lower-to-apple_gpu-full %s \
// RUN:   | FileCheck %s --check-prefix=GPU

// CPU-LABEL: func.func @chol_e2e
// CPU: %[[R:.*]] = tessera_apple.cpu.call
// CPU-SAME: abi = "lapack_spotrf"
// CPU-SAME: status = "executable"
// CPU-SAME: symbol = "tessera_apple_cpu_cholesky_f32"
// CPU: return %[[R]]
// CPU-NOT: ub.poison
// CPU-NOT: tile.cholesky
// CPU-NOT: = tessera.cholesky %

// GPU-LABEL: func.func @chol_e2e
// GPU: %[[RG:.*]] = tessera_apple.gpu.kernel_call
// GPU-SAME: status = "executable"
// GPU-SAME: symbol = "tessera_apple_gpu_cholesky_f32"
// GPU: return %[[RG]]
// GPU-NOT: ub.poison
// GPU-NOT: tile.cholesky
// GPU-NOT: = tessera.cholesky %
func.func @chol_e2e(%a: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = tessera.cholesky %a : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
