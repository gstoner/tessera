// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(tessera-lower-to-rocm)' %s | FileCheck %s

// Generic target legalization keeps an orphan dtype request logical: parser or
// host-conversion evidence is not a physical consumer. The architecture-owned
// ROCm WMMA generator still consumes its registered signed-INT4 contract.

module {
  func.func @target_dtype_defaults() {
    "test.dtype_request"() {numeric_policy = {storage = "int4"}}
        : () -> ()
    return
  }

  "tessera_rocm.wmma_gemm"() {
    name = "default_int4_wmma",
    m = 16 : i64, n = 16 : i64, k = 16 : i64,
    mt = 16 : i64, nt = 16 : i64,
    numeric_policy = {storage = "int4"}
  } : () -> ()
}

// CHECK: "test.dtype_request"
// CHECK-SAME: numeric_policy = {accum = "int32", storage = "int4"}
// CHECK-NOT: tessera.storage_container
// CHECK-NOT: tessera.storage_pack
// CHECK-NOT: tessera.storage_packed
// CHECK: gpu.module @default_int4_wmma_mod
// CHECK: gpu.func @default_int4_wmma
// CHECK: vector<2xi32>
