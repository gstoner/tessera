// Apple Target IR — artifact-mode husk + GPU symbol gating (review R1/R2/R3),
// and the value-mode no-poison contract (Apple Value Target IR sprint).
//
// Two distinct lowering intents are exercised on the SAME bare Tile IR:
//
//  * ARTIFACT mode (tessera-lower-to-apple_{cpu,gpu}) — inspection/dashboard
//    projection.  The value-less artifact op is emitted and the Tile op's used
//    results are replaced by `ub.poison` husks (R1: every result, incl.
//    multi-result svd; R2: not a misleading operand rebind).  GPU emits a
//    `symbol` ONLY for metal_runtime ops (R3): tri_solve yes; svd artifact_only,
//    no symbol.
//
//  * VALUE mode (tessera-lower-to-apple_{cpu,gpu}-full) — semantics-preserving.
//    Value-producing cpu.call / gpu.kernel_call ops carry real results; the
//    module must contain NO ub.poison and NO surviving tile.*.
//
// RUN: tessera-opt -tessera-lower-to-apple_cpu --allow-unregistered-dialect %s \
// RUN:   | FileCheck %s --check-prefix=ARTCPU
// RUN: tessera-opt -tessera-lower-to-apple_gpu --allow-unregistered-dialect %s \
// RUN:   | FileCheck %s --check-prefix=ARTGPU

// ARTCPU-LABEL: func.func @art
// ARTCPU: tessera_apple.cpu.vector_op
// ARTCPU-SAME: symbol = "tessera_apple_cpu_svd_f32"
// ARTCPU: ub.poison
// ARTCPU: ub.poison
// ARTCPU: ub.poison
// ARTCPU-NOT: tile.svd

// ARTGPU-LABEL: func.func @art
// ARTGPU: tessera_apple.gpu.metal_kernel
// ARTGPU-SAME: status = "artifact_only"
// ARTGPU-NOT: symbol =
func.func @art(%a: tensor<6x4xf32>)
    -> (tensor<6x4xf32>, tensor<4xf32>, tensor<4x4xf32>) {
  %u, %s, %v = "tile.svd"(%a) {source = "tessera.svd", result = "v0", ordinal = 0 : i64}
      : (tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<4xf32>, tensor<4x4xf32>)
  return %u, %s, %v : tensor<6x4xf32>, tensor<4xf32>, tensor<4x4xf32>
}
// REQUIRES: tessera-apple-backend
//
