// RUN: tessera-opt --tessera-autodiff-paired --tessera-ebm-canonicalize --tessera-ebm-lower-langevin --tessera-to-linalg --inline --convert-elementwise-to-linalg --canonicalize --cse "--tessera-row-program-to-gpu=backend=rocm entry=tessera_jit_ebm_langevin_loop" %s | FileCheck %s
// REQUIRES: tessera-ebm
//
// The sphere integrator as one cooperative kernel (2026-09-16, M1). Per step
// the kernel performs four ORDERED row reductions — the entry norm, the two
// tangent projections and the retraction norm — each a lane store into the
// shared row-reduction array, a barrier, a leader-side sequential fold and a
// broadcast, so a reduced quantity matches a sequential host fold rather than
// merely matching to a tolerance. The per-row status word rides the loop in a
// register beside the state and the key words, and the two singularities (an
// entry state off the sphere, a retraction underflow) are reported rather than
// repaired. 6 features -> 8 lanes.

module {
  func.func @energy(%y: tensor<3x6xf32>, %x: tensor<3x6xf32>) -> tensor<3xf32>
      attributes {tessera.autodiff = "reverse"} {
    %d = "tessera.sub"(%x, %y) : (tensor<3x6xf32>, tensor<3x6xf32>) -> tensor<3x6xf32>
    %sq = "tessera.mul"(%d, %d) : (tensor<3x6xf32>, tensor<3x6xf32>) -> tensor<3x6xf32>
    %s = "tessera.reduce"(%sq) {axis = 1 : i64, kind = "sum"} : (tensor<3x6xf32>) -> tensor<3xf32>
    %half = arith.constant dense<5.000000e-01> : tensor<3xf32>
    %e = "tessera.mul"(%s, %half) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    return %e : tensor<3xf32>
  }
  func.func @tessera_jit_ebm_langevin_loop(%y0: tensor<3x6xf32>, %x: tensor<3x6xf32>, %key0: tensor<2xi64>)
      -> (tensor<3x6xf32>, tensor<2xi64>, tensor<3xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %steps = arith.constant 2 : index
    %ok = arith.constant dense<0> : tensor<3xi32>
    %r:3 = scf.for %t = %c0 to %steps step %c1 iter_args(%y = %y0, %key = %key0, %status = %ok)
        -> (tensor<3x6xf32>, tensor<2xi64>, tensor<3xi32>) {
      %n:3 = "tessera_ebm.langevin_step"(%y, %key, %x)
          { operandSegmentSizes = array<i32: 1, 1, 0, 1>, energy_fn = @energy, eta = 0.1 : f64, temperature = 0.4 : f64, manifold = "sphere" }
          : (tensor<3x6xf32>, tensor<2xi64>, tensor<3x6xf32>) -> (tensor<3x6xf32>, tensor<2xi64>, tensor<3xi32>)
      %acc = arith.ori %status, %n#2 : tensor<3xi32>
      scf.yield %n#0, %n#1, %acc : tensor<3x6xf32>, tensor<2xi64>, tensor<3xi32>
    }
    return %r#0, %r#1, %r#2 : tensor<3x6xf32>, tensor<2xi64>, tensor<3xi32>
  }
}

// CHECK: tessera.row_program.features = 6
// CHECK: tessera.row_program.rows = 3
// CHECK: llvm.mlir.global private @row_reduction() {addr_space = 3 : i32{{.*}} : !llvm.array<8 x f32>
// CHECK: gpu.func @row_program(%{{.*}}: !llvm.ptr<1>, %{{.*}}: !llvm.ptr<1>, %{{.*}}: !llvm.ptr<1>, %{{.*}}: !llvm.ptr<1>, %{{.*}}: !llvm.ptr<1>, %{{.*}}: !llvm.ptr<1>, %{{.*}}: index) kernel attributes {known_block_size = array<i32: 8, 1, 1>}
// The K-step loop carries the state, both key words and the status word in registers.
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (f32, i64, i64, i32)
// One ordered reduction: lane store, barrier, leader fold over the features, broadcast.
// CHECK: llvm.store {{.*}} : f32, !llvm.ptr<3>
// CHECK: gpu.barrier
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (f32)
// CHECK: llvm.load {{.*}} : !llvm.ptr<3> -> f32
// CHECK: arith.addf
// CHECK: gpu.barrier
// The entry precondition and the retraction underflow become status bits.
// CHECK: math.absf
// CHECK: arith.cmpf ogt
// CHECK: arith.ori
// CHECK: gpu.return
// CHECK-NOT: linalg.
// CHECK-NOT: tensor.
// CHECK-NOT: tessera_ebm.
