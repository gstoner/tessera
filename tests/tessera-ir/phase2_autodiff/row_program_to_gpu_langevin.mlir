// RUN: tessera-opt --tessera-autodiff-paired --tessera-ebm-canonicalize --tessera-ebm-lower-langevin --tessera-to-linalg --inline --convert-elementwise-to-linalg --canonicalize --cse "--tessera-row-program-to-gpu=backend=nvidia entry=loop" %s | FileCheck %s
// RUN: not tessera-opt --tessera-autodiff-paired --tessera-ebm-canonicalize --tessera-ebm-lower-langevin --tessera-to-linalg --convert-elementwise-to-linalg --canonicalize --cse "--tessera-row-program-to-gpu=backend=nvidia entry=loop" %s 2>&1 | FileCheck %s --check-prefix=CALL
// REQUIRES: tessera-ebm
//
// The EBM Langevin loop through the compiler alone (2026-09-16): the paired
// autodiff pass derives the gradient of the quadratic energy, the EBM lowering
// emits the step with Philox-4x32-10 / Box-Muller noise, tessera-to-linalg
// and inlining leave a [rows, features] row program, and the row-program
// emitter folds it into ONE cooperative gpu.func: one block per row, one lane
// per feature (8 features -> 8 lanes), the three-step scf.for carrying the
// state and both key words in registers, guarded loads and stores over
// !llvm.ptr<1> arguments, and no linalg/tensor op left. The second RUN skips
// --inline: the emitter refuses a call with a diagnostic instead of guessing.

module {
  func.func @energy(%y: tensor<4x8xf32>, %x: tensor<4x8xf32>) -> tensor<4xf32>
      attributes {tessera.autodiff = "reverse"} {
    %d = "tessera.sub"(%x, %y) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
    %sq = "tessera.mul"(%d, %d) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
    %s = "tessera.reduce"(%sq) {axis = 1 : i64, kind = "sum"} : (tensor<4x8xf32>) -> tensor<4xf32>
    %half = arith.constant dense<5.000000e-01> : tensor<4xf32>
    %e = "tessera.mul"(%s, %half) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %e : tensor<4xf32>
  }
  func.func @loop(%y0: tensor<4x8xf32>, %x: tensor<4x8xf32>, %key0: tensor<2xi64>) -> (tensor<4x8xf32>, tensor<2xi64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %steps = arith.constant 3 : index
    %r:2 = scf.for %t = %c0 to %steps step %c1 iter_args(%y = %y0, %key = %key0) -> (tensor<4x8xf32>, tensor<2xi64>) {
      %n:2 = "tessera_ebm.langevin_step"(%y, %key, %x)
          { energy_fn = @energy, eta = 0.1 : f64, temperature = 0.7 : f64, manifold = "euclidean" }
          : (tensor<4x8xf32>, tensor<2xi64>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<2xi64>)
      scf.yield %n#0, %n#1 : tensor<4x8xf32>, tensor<2xi64>
    }
    return %r#0, %r#1 : tensor<4x8xf32>, tensor<2xi64>
  }
}

// CHECK: module attributes {
// CHECK-SAME: tessera.row_program.backend = "nvidia"
// CHECK-SAME: tessera.row_program.entry = "loop"
// CHECK-SAME: tessera.row_program.features = 8
// CHECK-SAME: tessera.row_program.rows = 4
// CHECK: gpu.module @native_row
// CHECK: llvm.mlir.global private @row_reduction
// CHECK: gpu.func @row_program(%{{.*}}: !llvm.ptr<1>, %{{.*}}: !llvm.ptr<1>, %{{.*}}: !llvm.ptr<1>, %{{.*}}: !llvm.ptr<1>, %{{.*}}: !llvm.ptr<1>, %{{.*}}: index) kernel attributes {known_block_size = array<i32: 8, 1, 1>}
// CHECK: gpu.block_id
// CHECK: gpu.thread_id
// CHECK: scf.if
// CHECK: llvm.load
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (f32, i64, i64)
// CHECK: arith.mului_extended
// CHECK: math.log
// CHECK: math.cos
// CHECK: scf.yield {{.*}} : f32, i64, i64
// CHECK: llvm.store
// CHECK: gpu.return
// CHECK-NOT: linalg.
// CHECK-NOT: tensor.
// CHECK-NOT: tessera_ebm.

// CALL: inline every call before lowering
