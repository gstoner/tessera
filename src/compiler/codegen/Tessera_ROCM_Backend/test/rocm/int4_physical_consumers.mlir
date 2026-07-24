// RUN: %trop %s --allow-unregistered-dialect --generate-rocm-int4-pack-kernel | FileCheck %s

module {
  "tessera_rocm.int4_pack"() {
    name = "packed_relu", kind = "relu"
  } : () -> ()
  "tessera_rocm.int4_pack"() {
    name = "packed_sparse_gather", kind = "sparse_gather"
  } : () -> ()
  "tessera_rocm.int4_pack"() {
    name = "packed_cache_append", kind = "cache_append"
  } : () -> ()
}

// CHECK-LABEL: gpu.func @packed_relu(
// CHECK-SAME: memref<?xi8>, %{{.*}}: memref<?xi8>, %{{.*}}: index
// CHECK: arith.cmpi sgt
// CHECK: arith.select
// CHECK: memref.store

// CHECK-LABEL: gpu.func @packed_sparse_gather(
// CHECK-SAME: memref<?xi8>, %{{.*}}: memref<?xi64>, %{{.*}}: memref<?xi8>
// CHECK: memref.load
// CHECK: arith.index_cast
// CHECK: arith.shrui
// CHECK: memref.store

// CHECK-LABEL: gpu.func @packed_cache_append(
// CHECK-SAME: memref<?xi8>, %{{.*}}: memref<?xi8>, %{{.*}}: index, %{{.*}}: index
// CHECK: memref.load
// CHECK: arith.addi
// CHECK: memref.store
