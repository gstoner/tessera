// SD1 (tree multi-path rejection) — the ROCm kernel-gen structure for
// tessera.spec_accept_tree_sample: thread/path Leviathan match-length (exp +
// cmpf ole) into LDS, barrier, thread-0 argmax (cmpi sgt + select), two OUT
// stores [path_idx, prefix_length].
//
// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt %s --generate-rocm-spec-accept-tree-sample-kernel | FileCheck %s

// CHECK-LABEL: func.func @f
// CHECK:       gpu.func @tessera_spec_accept_tree_sample_0(%{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: memref<?xi32>) workgroup({{.*}}memref<256xi32, #gpu.address_space<workgroup>>) kernel
// CHECK:         scf.for
// CHECK:           math.exp
// CHECK:           arith.cmpf ole
// CHECK:         gpu.barrier
// CHECK:         arith.cmpi sgt
// CHECK:         memref.store
// CHECK:         gpu.return
func.func @f(%t: tensor<3x4xf32>, %d: tensor<3x4xf32>, %u: tensor<3x4xf32>) -> tensor<2xi32> {
  %r = "tessera.spec_accept_tree_sample"(%t, %d, %u) : (tensor<3x4xf32>, tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
