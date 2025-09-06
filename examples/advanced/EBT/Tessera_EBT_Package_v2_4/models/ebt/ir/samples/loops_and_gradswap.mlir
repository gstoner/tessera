// RUN: tessera-ebt-opt %s --ebt-K=3 --ebt-T=2 --ebt-use-jvp=true | FileCheck %s
module {
  // EBT driver (sketch): encode→init→inner loop
  func.func @driver(%x: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %h = "tessera.encode"(%x) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %y = "tessera.ebt.decode_init"(%x) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %g = "tessera.autodiff.grad_y"(%h, %y) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %y2 = "tessera.ebt.inner_step"(%y, %g) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    return %y2 : tensor<?x?x?xf32>
  }
}

// CHECK: // tessera-ebt-opt options: K=3 T=2 useJVP=true
// CHECK: // passes: tessera-ebt-materialize-loops ; tessera-ebt-select-grad-path
// CHECK: scf.for
// CHECK: scf.for
// CHECK: ebt.energy_bilinear_jvp
