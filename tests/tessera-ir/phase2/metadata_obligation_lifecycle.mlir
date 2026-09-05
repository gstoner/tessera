// RUN: tessera-opt %s --tessera-record-metadata --tessera-verify-metadata-obligation | FileCheck %s
// The first record must verify and retire the frontend boundary, then establish
// the next boundary's snapshot. No stale frontier exception may survive.
// CHECK: tessera.metadata_snapshot = {f = {target =
// CHECK-NOT: dim_names
// CHECK-NOT: tessera.lowering.dropped
module attributes {tessera.metadata_snapshot = {f = {dim_names = ["[M]", 1 : i64]}}} {
  func.func @f(%a: tensor<8xf32>) -> tensor<8xf32> attributes {
    tessera.target = "nvidia_sm90",
    tessera.lowering.dropped = {dim_names = "not_yet_carried:FRONTEND-IR-MEDIUM-1"}
  } {
    return %a : tensor<8xf32>
  }
}
