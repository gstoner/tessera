// L-series linalg pilot — L4: tile.cholesky → tessera_apple Target IR.
//
// The Tile→Apple passes lower the opaque `tile.cholesky` (carrying
// source = "tessera.cholesky") to a registered tessera_apple target op that
// names the runtime C ABI `symbol` the executor (L6) invokes:
//   CPU → tessera_apple.cpu.vector_op {abi="lapack_spotrf", symbol=...}
//   GPU → tessera_apple.gpu.metal_kernel {status="metal_runtime", symbol=...}
//
// RUN: tessera-opt -tessera-lower-to-apple_cpu --allow-unregistered-dialect %s \
// RUN:   | FileCheck %s --check-prefix=CPU
// RUN: tessera-opt -tessera-lower-to-apple_gpu --allow-unregistered-dialect %s \
// RUN:   | FileCheck %s --check-prefix=GPU

// CPU: tessera_apple.cpu.vector_op
// CPU-SAME: abi = "lapack_spotrf"
// CPU-SAME: op_kind = "cholesky"
// CPU-SAME: symbol = "tessera_apple_cpu_cholesky_f32"

// GPU: tessera_apple.gpu.metal_kernel
// GPU-SAME: kernel = "cholesky_contract"
// GPU-SAME: status = "metal_runtime"
// GPU-SAME: symbol = "tessera_apple_gpu_cholesky_f32"
module {
  "tile.cholesky"() {source = "tessera.cholesky", result = "v0", ordinal = 0 : i64, lower = true} : () -> ()
}
// REQUIRES: tessera-apple-backend
//
