// REQUIRES: tessera-apple-backend
//
// RUN: tessera-opt %s --tessera-control-while-to-apple_gpu | FileCheck %s

// Phase-G close-out D — lower the Graph-IR bounded while `tessera.control_while`
// to the Apple Target-IR op `tessera_apple.gpu.control_while` (value-preserving;
// records the run_graph_while runtime symbol). IR-only (Decision #19 hardware-free
// layer). Body + cond are symbol-referenced func.funcs (not regions).
func.func private @while_body(%c: tensor<1x4xf32>) -> tensor<1x4xf32>
func.func private @while_cond(%c: tensor<1x4xf32>) -> tensor<1x4xf32>

// CHECK-LABEL: func.func @bounded_while
func.func @bounded_while(%init: tensor<1x4xf32>, %w: tensor<4x4xf32>,
                         %thr: tensor<1xf32>) -> tensor<1x4xf32> {
  // CHECK: %[[R:.*]] = tessera_apple.gpu.control_while %{{.*}} {
  // CHECK-SAME: body = @while_body
  // CHECK-SAME: carry_arg_index = 0 : i64
  // CHECK-SAME: cond = @while_cond
  // CHECK-SAME: max_iters = 3 : i64
  // CHECK-SAME: status = "artifact"
  // CHECK-SAME: symbol = "tessera_apple_gpu_run_graph_while_f32"
  // CHECK-NOT: tessera.control_while
  // CHECK: return %[[R]]
  %r = "tessera.control_while"(%init, %w, %thr) {
    body = @while_body, cond = @while_cond,
    carry_arg_index = 0 : i64, max_iters = 3 : i64
  } : (tensor<1x4xf32>, tensor<4x4xf32>, tensor<1xf32>) -> tensor<1x4xf32>
  return %r : tensor<1x4xf32>
}
