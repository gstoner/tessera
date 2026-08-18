// RUN: tessera-opt --pass-pipeline='builtin.module(tessera-graph-to-schedule,tessera-schedule-to-tile,tessera-x86-executable{family=tridiagonal_solve input=tile output=target arch=x86_64_avx512})' %s | FileCheck %s

module attributes {tessera.target = "x86", tessera.arch = "zen5-avx512"} {
  func.func @tridiagonal(%dl: tensor<8xf32>, %d: tensor<8xf32>,
      %du: tensor<8xf32>, %rhs: tensor<17x8xf32>) -> tensor<17x8xf32> {
    %result = tessera.tridiagonal_solve %dl, %d, %du, %rhs
      : (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<17x8xf32>)
        -> tensor<17x8xf32>
    return %result : tensor<17x8xf32>
  }
}

// CHECK: tessera.pipeline.family = "tridiagonal_solve"
// CHECK: tessera.pipeline.target_ir_consumer = "tessera_x86"
// CHECK: tessera_x86.abi_call
// CHECK-SAME: symbol = "tessera_x86_avx512_tridiagonal_solve_f32"
// CHECK: call @tessera_x86_avx512_tridiagonal_solve_f32
// CHECK: cf.assert {{.*}}, "x86 tridiagonal solve rejected an invalid or singular system"
// CHECK-NOT: tile.tridiagonal_solve_kernel
// CHECK-NOT: schedule.tridiagonal_solve
// REQUIRES: tessera-x86-target-ir
