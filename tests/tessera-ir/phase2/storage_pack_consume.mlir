// RUN: tessera-opt --allow-unregistered-dialect --tessera-storage-pack-consume -split-input-file -verify-diagnostics %s | FileCheck %s
//
// C4 part 1 (2026-06-23): the first real *consumer* of the C4 packing markers
// (tessera.storage_packed / tessera.storage_container) — turns them from inert
// annotations into a concrete #tile.packed_format descriptor a backend's
// packed load/store reads. elements_per_container = container_bits /
// storage_bits while logical_bits remains explicit (notably FP6-in-i8).

// fp4 (4-bit) packs 2 per int8 container.
// CHECK-LABEL: func.func @fp4
// CHECK: tessera.storage_pack = #tile.packed_format<logical = "fp4_e2m1", container = "int8", logical_bits = 4, elements_per_container = 2, signedness = "format_defined", encoding = "e2m1", lane_order = "low_to_high">
func.func @fp4(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %c = "tessera.matmul"(%a, %a) {numeric_policy = {storage = "fp4_e2m1", accum = "fp32"}, tessera.storage_packed = true, tessera.storage_container = "int8"}
       : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %c : tensor<4x4xf32>
}

// -----

// fp6 (6-bit) only fits 1 per int8 container (2 bits slack).
// CHECK-LABEL: func.func @fp6
// CHECK: tessera.storage_pack = #tile.packed_format<logical = "fp6_e3m2", container = "int8", logical_bits = 6, elements_per_container = 1, signedness = "format_defined", encoding = "e3m2", lane_order = "scalar_lsb">
func.func @fp6(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %c = "tessera.matmul"(%a, %a) {numeric_policy = {storage = "fp6_e3m2", accum = "fp32"}, tessera.storage_packed = true, tessera.storage_container = "int8"}
       : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %c : tensor<4x4xf32>
}

// -----

// int4 packs 2 per int8 container (the AMD IU4 path).
// CHECK-LABEL: func.func @int4
// CHECK: tessera.storage_pack = #tile.packed_format<logical = "int4", container = "int8", logical_bits = 4, elements_per_container = 2, signedness = "signed_twos_complement", encoding = "twos_complement", lane_order = "low_to_high">
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
