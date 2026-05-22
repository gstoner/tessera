// RUN: mlir-opt %s -tessera-ebt-canonicalize -tessera-ebt-materialize-loops="K=2 T=3" -tessera-ebt-select-grad-path=true | FileCheck %s
module {
  func.func @driver(%x: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %h = "tessera.encode"(%x) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %y = "tessera.ebt.decode_init"(%x) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %g = "tessera.autodiff.grad_y"(%h, %y) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %y2 = "tessera.ebt.inner_step"(%y, %g) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    return %y2 : tensor<?x?x?xf32>
  }
}

// CHECK: scf.for
// CHECK: scf.for
// CHECK: ebt.energy_bilinear_jvp
