// CF4c-cont — GenerateROCMControlForKernel lowers an elementwise-body
// tessera.control_while to ONE gpu.func device kernel: grid over the carry
// elements; per thread, a bounded scf.while over (counter, carry):
//   while (i < max_iters AND cond_scalar(c) > 0) { c = body_scalar(c); i++ }.
// The cond is SHORT-CIRCUITED behind the i < max_iters bound (evaluated only
// inside an scf.if), so @cond never runs past the bound — matching CF2c.
// (On-device execution on gfx1151 is proven by
// tests/unit/test_rocm_control_while_exec.py.)
//
// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt %s -split-input-file --generate-rocm-control-for-kernel \
// RUN:   --allow-unregistered-dialect | FileCheck %s

// ─── body = add(c, c) (doubling), cond = relu(c) (> 0 iff c > 0) ─────────────
func.func @wb(%c: tensor<8xf32>) -> tensor<8xf32> {
  %0 = "tessera.add"(%c, %c) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
func.func @wc(%c: tensor<8xf32>) -> tensor<8xf32> {
  %0 = "tessera.relu"(%c) : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK:       gpu.func @tessera_control_while_0(%{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: index) kernel
// CHECK:         %[[X:.*]] = memref.load
// CHECK:         %[[W:.*]]:2 = scf.while (%[[I:.*]] = %{{.*}}, %[[C:.*]] = %[[X]]) : (index, f32) -> (index, f32)
// CHECK:           %[[LT:.*]] = arith.cmpi ult, %[[I]]
// CHECK:           %[[CONT:.*]] = scf.if %[[LT]] -> (i1) {
// CHECK:             arith.maximumf %[[C]]
// CHECK:             arith.cmpf ogt
// CHECK:             scf.yield
// CHECK:           } else {
// CHECK:             arith.constant false
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           scf.condition(%[[CONT]]) %[[I]], %[[C]]
// CHECK:         } do {
// CHECK:           arith.addf
// CHECK:           arith.addi
// CHECK:           scf.yield
// CHECK:         }
// CHECK:         memref.store %[[W]]#1
// CHECK:         gpu.return
func.func @f(%init: tensor<8xf32>) -> tensor<8xf32> {
  %r = "tessera.control_while"(%init) {
    body = @wb, cond = @wc, carry_arg_index = 0 : i64, max_iters = 4 : i64
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %r : tensor<8xf32>
}

// -----
// ─── a non-elementwise body (matmul) is NOT lowered — left for the guard ────
func.func @mb(%c: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = "tessera.matmul"(%c, %c) : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
func.func @mc(%c: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = "tessera.relu"(%c) : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func.func @g
// CHECK:       tessera.control_while
// CHECK-NOT:   gpu.func
func.func @g(%init: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %r = "tessera.control_while"(%init) {
    body = @mb, cond = @mc, carry_arg_index = 0 : i64, max_iters = 4 : i64
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %r : tensor<8x8xf32>
}
