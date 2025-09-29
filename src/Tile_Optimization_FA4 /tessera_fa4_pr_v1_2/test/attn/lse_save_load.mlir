// RUN: tessera-opt %s -tessera-lower-attn | FileCheck %s
%lse = "tessera.attn.lse.save"(%scores) : (memref<128x64xf32>) -> memref<128xf32>
%use = "tessera.attn.lse.load"() : () -> memref<128xf32>
// CHECK: lse
