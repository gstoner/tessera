// RUN: tessera-opt %s -tpp-legalize-space-time -tpp-halo-infer -tpp-fuse-stencil-time | FileCheck %s
//
// FuseStencilTime groups sibling stencils that read the same field (the
// canonical %Hx/%Hy shallow-water case).  Both directional gradients on %h
// join one fusion group whose halo is the union of theirs: [1,0] U [0,1] =
// [1,1].

func.func @grads(%h: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>) {
  %hx = "tpp.grad"(%h) { axis = 0 : i64 } : (tensor<64x64xf32>) -> tensor<64x64xf32>
  %hy = "tpp.grad"(%h) { axis = 1 : i64 } : (tensor<64x64xf32>) -> tensor<64x64xf32>
  // CHECK: tpp.grad
  // CHECK-SAME: tpp.fuse.group = 0
  // CHECK-SAME: tpp.fuse.halo = [1, 1]
  // CHECK-SAME: tpp.fuse.members = 2
  // CHECK: tpp.grad
  // CHECK-SAME: tpp.fuse.group = 0
  // CHECK-SAME: tpp.fuse.halo = [1, 1]
  // CHECK-SAME: tpp.fuse.members = 2
  return %hx, %hy : tensor<64x64xf32>, tensor<64x64xf32>
}
