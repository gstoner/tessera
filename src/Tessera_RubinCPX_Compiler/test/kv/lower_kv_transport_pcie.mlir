
// RUN: %cpx_opt %s -tessera-lower-kv-transport | FileCheck %s
// CHECK: module
module {
  // skeleton: ensure pass runs without errors on kv ops.
  "tessera.target.cpx.kv.export"() : () -> ()
  "tessera.target.cpx.kv.import"() : () -> ()
  "tessera.target.cpx.kv.prefetch"() : () -> ()
}
