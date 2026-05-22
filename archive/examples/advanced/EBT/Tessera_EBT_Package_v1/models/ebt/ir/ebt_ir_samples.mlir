// RUN: mlir-opt %s -tessera-ebt-canonicalize -tessera-ebt-lower | FileCheck %s

// Encode context
%h = "tessera.encode"(%x) : (tensor<BxLxD>) -> tensor<BxLxD>

// K candidates init
%y0 = "tessera.ebt.decode_init"(%x) : (tensor<BxLxD>) -> tensor<BxKxLxD>

// CHECK: scf.for
// CHECK: ebt.grad_y
// CHECK: ebt.inner_step
// CHECK: ebt.energy
// CHECK: ebt.self_verify
