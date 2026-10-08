// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --generate-wmma-gemm-kernel='via-tile=true' %s | FileCheck %s
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --generate-wmma-gemm-kernel='via-tile=true' --lower-tile-to-rocm='arch=gfx1201' %s -o /dev/null

// The Schedule selects a 128x64 eight-wave tile. K32 staging has only
// 128 RHS vectors for 256 threads: half the threads must not dereference
// global/LDS addresses beyond the slab, but all must meet its barriers.
module attributes {tessera.target = "rocm", tessera.arch = "gfx1201"} {
  func.func @partial_copy(%a: tensor<128x128xf8E4M3FN>,
                          %b: tensor<4096x128xf8E4M3FN>,
                          %sa: tensor<128x4xf32>,
                          %sb: tensor<4x4096xf32>) -> tensor<128x4096xf32> {
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {
      numeric_policy = {accum = "fp32", execution_mode = "exact_per_block"},
      scale_layout = {granularity = "block", block = [1, 32], format = "fp32"},
      transposeB = true
    } : (tensor<128x128xf8E4M3FN>, tensor<4096x128xf8E4M3FN>,
         tensor<128x4xf32>, tensor<4x4096xf32>) -> tensor<128x4096xf32>
    return %0 : tensor<128x4096xf32>
  }
}
// CHECK: arith.cmpi ult
// CHECK: scf.if {{.*}} -> (vector<16xf8E4M3FN>)
// CHECK: vector.load
// CHECK: scf.yield
// CHECK: } else {
// CHECK: arith.constant dense<0.000000e+00> : vector<16xf8E4M3FN>
// CHECK: scf.yield
// CHECK: scf.if
// CHECK: vector.store
// CHECK: gpu.barrier
