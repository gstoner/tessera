// RUN: mlir-opt %s -tessera-autodiff=mode=vjp,targets=ebt.energy_mlp | FileCheck %s
module attributes {tessera.autodiff.targets = ["ebt.energy_mlp"]} {
  // Primal MLP energy
  func.func @energy_mlp_primal(%h: tensor<?x?x?xf32>, %y: tensor<?x?x?xf32>,
                               %W1: tensor<?x?xf32>, %b1: tensor<?xf32>,
                               %W2: tensor<?x?xf32>, %b2: tensor<?xf32>)
      -> tensor<?x1xf32> attributes { tessera.ebt.energy_mlp } {
    %z = "tessera.tile.concat"(%y, %h) {axis=-1}
        : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %u = "tessera.tile.linear"(%z, %W1, %b1)
        : (tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<?xf32>) -> tensor<?x?x?xf32>
    %a = "tessera.tile.gelu"(%u) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %o = "tessera.tile.linear"(%a, %W2, %b2)
        : (tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<?xf32>) -> tensor<?x?x1xf32>
    %E = "tessera.tile.reduce_sum"(%o) {dims=[1]}
        : (tensor<?x?x1xf32>) -> tensor<?x1xf32>
    return %E : tensor<?x1xf32>
  }

  // grad_y via autodiff driver (logical op)
  func.func @grad_y(%h: tensor<?x?x?xf32>, %y: tensor<?x?x?xf32>,
                    %W1: tensor<?x?xf32>, %b1: tensor<?xf32>,
                    %W2: tensor<?x?xf32>, %b2: tensor<?xf32>)
      -> tensor<?x?x?xf32> {
    %g = "tessera.autodiff.grad_y"(@energy_mlp_primal, %h, %y, %W1, %b1, %W2, %b2)
         : (!tosa.func, tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?xf32>)
           -> tensor<?x?x?xf32>
    return %g : tensor<?x?x?xf32>
  }
}

// CHECK-LABEL: func @grad_y
// CHECK: tessera.autodiff.grad_y
