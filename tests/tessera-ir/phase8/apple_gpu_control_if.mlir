// REQUIRES: tessera-apple-backend
//
// RUN: tessera-opt %s --tessera-control-if-to-apple_gpu | FileCheck %s

// Phase-G close-out C — lower the Graph-IR divergent if/else `tessera.control_if`
// to the Apple Target-IR op `tessera_apple.gpu.control_if` (value-preserving;
// records the run_graph_cond runtime symbol). IR-only (Decision #19 hardware-free
// layer). Both branch bodies are symbol-referenced func.funcs (not regions).
func.func private @then_body(%x: tensor<1x8xf32>) -> tensor<1x8xf32>
func.func private @else_body(%x: tensor<1x8xf32>) -> tensor<1x8xf32>

// CHECK-LABEL: func.func @divergent_if
func.func @divergent_if(%flag: tensor<1xf32>, %x: tensor<1x8xf32>,
                        %w: tensor<8x8xf32>) -> tensor<1x8xf32> {
  // CHECK: %[[R:.*]] = tessera_apple.gpu.control_if %{{.*}} {
  // CHECK-SAME: else_branch = @else_body
  // CHECK-SAME: flag_arg_index = 0 : i64
  // CHECK-SAME: status = "artifact"
  // CHECK-SAME: symbol = "tessera_apple_gpu_run_graph_cond_f32"
  // CHECK-SAME: then_branch = @then_body
  // CHECK-NOT: tessera.control_if
  // CHECK: return %[[R]]
  %r = "tessera.control_if"(%flag, %x, %w) {
    then_branch = @then_body, else_branch = @else_body,
    flag_arg_index = 0 : i64, out_shape = array<i64: 1, 8>
  } : (tensor<1xf32>, tensor<1x8xf32>, tensor<8x8xf32>) -> tensor<1x8xf32>
  return %r : tensor<1x8xf32>
}
