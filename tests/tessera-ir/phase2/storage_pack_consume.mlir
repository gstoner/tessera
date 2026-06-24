// RUN: tessera-opt --allow-unregistered-dialect --tessera-storage-pack-consume -split-input-file -verify-diagnostics %s | FileCheck %s
//
// C4 part 1 (2026-06-23): the first real *consumer* of the C4 packing markers
// (tessera.storage_packed / tessera.storage_container) — turns them from inert
// annotations into a concrete tessera.storage_pack = {logical, container,
// factor} descriptor a backend's packed load/store reads. factor =
// container_bits / storage_bits. The packed memory codegen + flipping
// legalize-dtypes default-on are the hardware-gated tail.

// fp4 (4-bit) packs 2 per int8 container.
// CHECK-LABEL: func.func @fp4
// CHECK: tessera.storage_pack = {container = "int8", factor = 2 : i64, logical = "fp4_e2m1"}
func.func @fp4(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %c = "tessera.matmul"(%a, %a) {numeric_policy = {storage = "fp4_e2m1", accum = "fp32"}, tessera.storage_packed = true, tessera.storage_container = "int8"}
       : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %c : tensor<4x4xf32>
}

// -----

// fp6 (6-bit) only fits 1 per int8 container (2 bits slack).
// CHECK-LABEL: func.func @fp6
// CHECK: tessera.storage_pack = {container = "int8", factor = 1 : i64, logical = "fp6_e3m2"}
func.func @fp6(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %c = "tessera.matmul"(%a, %a) {numeric_policy = {storage = "fp6_e3m2", accum = "fp32"}, tessera.storage_packed = true, tessera.storage_container = "int8"}
       : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %c : tensor<4x4xf32>
}

// -----

// int4 packs 2 per int8 container (the AMD IU4 path).
// CHECK-LABEL: func.func @int4
// CHECK: factor = 2 : i64, logical = "int4"
func.func @int4(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %c = "tessera.matmul"(%a, %a) {numeric_policy = {storage = "int4", accum = "int32"}, tessera.storage_packed = true, tessera.storage_container = "int8"}
       : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %c : tensor<4x4xf32>
}

// -----

// A storage wider than the container can't pack.
func.func @bad_widths(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // expected-error @+1 {{DTYPE_PACK_BAD_WIDTHS}}
  %c = "tessera.matmul"(%a, %a) {numeric_policy = {storage = "int32"}, tessera.storage_packed = true, tessera.storage_container = "int8"}
       : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %c : tensor<4x4xf32>
}
