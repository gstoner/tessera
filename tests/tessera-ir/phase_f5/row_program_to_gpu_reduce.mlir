// RUN: tessera-opt "--tessera-row-program-to-gpu=backend=rocm entry=row_normalize" %s | FileCheck %s
// RUN: not tessera-opt "--tessera-row-program-to-gpu=backend=rocm entry=missing" %s 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: not tessera-opt "--tessera-row-program-to-gpu=backend=rocm entry=too_wide" %s 2>&1 | FileCheck %s --check-prefix=WIDE
//
// The row-program emitter's reduction path (2026-09-16): a feature-axis
// linalg.reduce becomes an ordered shared-memory fold — every lane writes its
// element into @row_reduction (address space 3), a barrier, the block leader
// folds the combiner over the lanes in index order (the declared reduction
// order, matched by a sequential f32 fold on the host), a barrier, the result
// is broadcast through the same array, a barrier. 6 features -> 8 lanes with
// the two spare lanes masked. Programs outside the envelope fail closed.

#map = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0, 0)>
module {
  func.func @row_normalize(%x: tensor<3x6xf32>) -> tensor<3x6xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %e = tensor.empty() : tensor<3x6xf32>
    %sq = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%x, %x : tensor<3x6xf32>, tensor<3x6xf32>) outs(%e : tensor<3x6xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %m = arith.mulf %a, %b : f32
      linalg.yield %m : f32
    } -> tensor<3x6xf32>
    %re = tensor.empty() : tensor<3xf32>
    %init = linalg.fill ins(%zero : f32) outs(%re : tensor<3xf32>) -> tensor<3xf32>
    %s = linalg.reduce ins(%sq : tensor<3x6xf32>) outs(%init : tensor<3xf32>) dimensions = [1]
      (%in: f32, %acc: f32) {
        %add = arith.addf %in, %acc : f32
        linalg.yield %add : f32
      }
    %n = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
        ins(%s : tensor<3xf32>) outs(%re : tensor<3xf32>) {
    ^bb0(%v: f32, %o: f32):
      %sq2 = math.sqrt %v : f32
      linalg.yield %sq2 : f32
    } -> tensor<3xf32>
    %nx = tensor.expand_shape %n [[0, 1]] output_shape [3, 1] : tensor<3xf32> into tensor<3x1xf32>
    %y = linalg.generic {indexing_maps = [#map, #row, #map], iterator_types = ["parallel", "parallel"]}
        ins(%x, %nx : tensor<3x6xf32>, tensor<3x1xf32>) outs(%e : tensor<3x6xf32>) {
    ^bb0(%a: f32, %d: f32, %o: f32):
      %q = arith.divf %a, %d : f32
      linalg.yield %q : f32
    } -> tensor<3x6xf32>
    return %y : tensor<3x6xf32>
  }
  func.func @too_wide(%x: tensor<2x2048xf32>) -> tensor<2x2048xf32> {
    return %x : tensor<2x2048xf32>
  }
}

// CHECK: tessera.row_program.features = 6
// CHECK: tessera.row_program.rows = 3
// CHECK: llvm.mlir.global private @row_reduction() {addr_space = 3 : i32{{.*}} : !llvm.array<8 x f32>
// CHECK: gpu.func @row_program(%{{.*}}: !llvm.ptr<1>, %{{.*}}: !llvm.ptr<1>, %{{.*}}: index) kernel attributes {known_block_size = array<i32: 8, 1, 1>}
// CHECK: llvm.mlir.addressof @row_reduction
// CHECK: arith.mulf
// CHECK: llvm.store {{.*}} : f32, !llvm.ptr<3>
// CHECK: gpu.barrier
// CHECK: scf.if
// CHECK: scf.for {{.*}} to %c6
// CHECK: llvm.load {{.*}} : !llvm.ptr<3> -> f32
// CHECK: arith.addf
// CHECK: llvm.store {{.*}} : f32, !llvm.ptr<3>
// CHECK: gpu.barrier
// CHECK: llvm.load {{.*}} : !llvm.ptr<3> -> f32
// CHECK: gpu.barrier
// CHECK: math.sqrt
// CHECK: arith.divf
// CHECK: llvm.store {{.*}} : f32, !llvm.ptr<1>
// CHECK: gpu.return
// CHECK-NOT: linalg.
// CHECK-NOT: tensor.

// MISSING: entry
// WIDE: 1024
