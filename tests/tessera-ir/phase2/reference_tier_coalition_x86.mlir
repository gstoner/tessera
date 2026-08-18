// RUN: tessera-opt --pass-pipeline='builtin.module(tessera-graph-to-schedule,tessera-schedule-to-tile,tessera-x86-executable{family=coalition_butterfly input=tile output=target arch=x86_64_avx512})' %s | FileCheck %s

module attributes {tessera.target = "x86", tessera.arch = "zen5-avx512"} {
  func.func @subset_zeta(%input: tensor<3x32xf32>) -> tensor<3x32xf32> {
    %result = tessera.game_subset_zeta %input
      : (tensor<3x32xf32>) -> tensor<3x32xf32>
    return %result : tensor<3x32xf32>
  }
}

// CHECK: tessera.pipeline.family = "coalition_butterfly"
// CHECK: tessera.pipeline.target_ir_consumer = "tessera_x86"
// CHECK: tessera_x86.abi_call
// CHECK-SAME: symbol = "tessera_x86_avx512_coalition_butterfly_f32"
// CHECK: call @tessera_x86_avx512_coalition_butterfly_f32
// CHECK: cf.assert {{.*}}, "x86 coalition butterfly rejected its runtime contract"
// CHECK-NOT: tile.coalition_butterfly_kernel
// CHECK-NOT: schedule.coalition_butterfly
// REQUIRES: tessera-x86-target-ir
