// RUN: tessera-ebt-opt %s -tessera-ebt-canonicalize -tessera-ebt-lower --backend=rocm --ebt-K=2 --ebt-T=3 | FileCheck %s
module attributes {tessera.target = "rocm"} {

// Enc/Init stubs
func.func @encode(%x: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %h = "tessera.encode"(%x) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %h : tensor<?x?x?xf32>
}
func.func @decode_init(%x: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %y0 = "tessera.ebt.decode_init"(%x) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %y0 : tensor<?x?x?xf32>
}
func.func @energy_bilinear(%h: tensor<?x?x?xf32>, %y: tensor<?x?x?xf32>, %W: tensor<?x?xf32>) -> tensor<?x1xf32> attributes { tessera.ebt.energy } {
  %t0 = "tessera.tile.matmul"(%y, %W) : (tensor<?x?x?xf32>, tensor<?x?xf32>) -> tensor<?x?x?xf32>
  %E_tok = "tessera.tile.batched_dot"(%t0, %h) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x1xf32>
  %E = "tessera.tile.reduce_sum"(%E_tok) {dims=[1]} : (tensor<?x?x1xf32>) -> tensor<?x1xf32>
  return %E : tensor<?x1xf32>
}
func.func @grad_y(%h: tensor<?x?x?xf32>, %y: tensor<?x?x?xf32>, %W: tensor<?x?xf32>) -> tensor<?x?x?xf32> {
  %g = "tessera.autodiff.grad_y"(%h, %y, %W) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?xf32>) -> tensor<?x?x?xf32>
  return %g : tensor<?x?x?xf32>
}

func.func @ebt_infer(%x: tensor<?x?x?xf32>, %W: tensor<?x?xf32>) -> tensor<?x?x?xf32> {
  %h = call @encode(%x) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %y = call @decode_init(%x) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %y : tensor<?x?x?xf32>
}
}

// CHECK: // tessera-ebt-opt options: K=2 T=3 useJVP=false backend=rocm
// CHECK: // pipelines: tessera-ebt-canonicalize | tessera-ebt-lower
// CHECK: tessera.target.rocm.mfma
