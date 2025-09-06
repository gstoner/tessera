// RUN: mlir-opt %s -tessera-autodiff=mode=vjp,targets=ebt.energy --split-input-file | FileCheck %s
module attributes {tessera.autodiff.targets = ["ebt.energy"]} {
  func.func @energy_primal(%h: tensor<?x?x?xf32>, %y: tensor<?x?x?xf32>,
                           %W1: tensor<?x?xf32>, %b1: tensor<?xf32>,
                           %W2: tensor<?x?xf32>, %b2: tensor<?xf32>) -> tensor<?x1xf32> attributes { tessera.ebt.energy } {
    %0 = call @ebt_energy_mlp(%h, %y, %W1, %b1, %W2, %b2) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?xf32>) -> tensor<?x1xf32>
    return %0 : tensor<?x1xf32>
  }

  func.func @grad_y(%h: tensor<?x?x?xf32>, %y: tensor<?x?x?xf32>,
                    %W1: tensor<?x?xf32>, %b1: tensor<?xf32>,
                    %W2: tensor<?x?xf32>, %b2: tensor<?xf32>) -> tensor<?x?x?xf32> {
    %g = tessera.autodiff.grad_y @energy_primal(%h, %y, %W1, %b1, %W2, %b2)
         : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?xf32>)
           -> tensor<?x?x?xf32>
    return %g : tensor<?x?x?xf32>
  }
}
// CHECK-LABEL: func @grad_y
// CHECK: tessera.autodiff.grad_y
