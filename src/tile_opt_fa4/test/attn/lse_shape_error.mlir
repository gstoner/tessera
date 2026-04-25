// RUN: not tessera-opt %s -tessera-lower-attn 2>&1 | FileCheck %s
%lse = "tessera.attn.lse.save"(%scores) : (memref<32x64xf32>) -> memref<16xf32>
// CHECK: lse length must equal scores rows
