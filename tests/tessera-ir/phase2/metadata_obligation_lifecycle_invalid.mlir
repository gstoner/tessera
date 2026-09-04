// RUN: not tessera-opt %s --tessera-record-metadata 2>&1 | FileCheck %s
// A new record must not erase an unacknowledged loss at the preceding boundary.
// CHECK: METADATA_OBLIGATION_SILENT_DROP
module attributes {tessera.metadata_snapshot = {f = {dim_names = ["[M]", 1 : i64]}}} {
  func.func @f(%a: tensor<8xf32>) -> tensor<8xf32> {
    return %a : tensor<8xf32>
  }
}
