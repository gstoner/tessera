
# RUN: %cpx_opt %s -tessera-lower-kv-transport | FileCheck %s

module {
  %0 = memref.alloc() : memref<1024xi8>
  %t = "tessera.target.cpx.kv.export"(%0, "nvlink", 33554432)
       : (memref<1024xi8>, !tessera.target.cpx<KVPolicyAttr>, i64) -> !async.token
  // CHECK: func.call @tessera_kv_send_nvlink
}
