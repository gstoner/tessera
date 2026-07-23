// RUN: tessera-opt --tessera-layout-assignment %s | FileCheck %s
// RUN: tessera-opt --tessera-layout-assignment --tessera-layout-legality %s | FileCheck %s --check-prefix=LEGAL
//
// LayoutAssignmentPass v1 (2026-06-17): seed kernel-producer layouts, propagate
// through pointwise, insert cast{layout} at consumer accept-set boundaries. The
// second RUN proves the assignment output is LEGAL (LayoutLegalityPass runs clean
// after it — assign + verify, the two-sided contract).

// SEED + PROPAGATE: matmul result is stamped row_major; the relu consuming it
// (pointwise) inherits row_major.
// CHECK-LABEL: func.func @seed_and_propagate
// CHECK: tessera.matmul
// CHECK-SAME: tessera.layout = "row_major"
// CHECK: tessera.relu
// CHECK-SAME: tessera.layout = "row_major"
// LEGAL-LABEL: func.func @seed_and_propagate
func.func @seed_and_propagate(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) -> tensor<4x16xf32> {
  %0 = "tessera.matmul"(%a, %b) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  %1 = "tessera.relu"(%0) : (tensor<4x16xf32>) -> tensor<4x16xf32>
  return %1 : tensor<4x16xf32>
}

// -----

// INSERT cast{layout}: the lhs producer is tagged "tile" (∉ matmul's accept-set
// {row_major, col_major}), so the pass splices a tessera.cast{layout="row_major"}
// marker before the matmul lhs. After assignment the program is legal (the LEGAL
// run is clean — the inserted cast's layout is in the accept-set, and "tile" is a
// canonical layout name so the original cast is legal too).
// CHECK-LABEL: func.func @insert_cast_at_boundary
// CHECK: tessera.cast
// CHECK-SAME: tessera.layout = "tile"
// CHECK: tessera.cast
// CHECK-SAME: tessera.layout = "row_major"
// CHECK: tessera.matmul
// LEGAL-LABEL: func.func @insert_cast_at_boundary
func.func @insert_cast_at_boundary(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) -> tensor<4x16xf32> {
  %at = "tessera.cast"(%a) {tessera.layout = "tile"} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %0 = "tessera.matmul"(%at, %b) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// -----

// Complete envelope: rank-2 transpose flips row-major to column-major, the
// pointwise epilogue retains its agreed storage descriptor, and the last-axis
// reduction receives one descriptor-preserving row-major materialization.
// CHECK-LABEL: func.func @transpose_packed_epilogue_reduce
// CHECK: tessera.transpose
// CHECK-SAME: tessera.layout = "col_major"
// CHECK-SAME: tessera.storage_packed = true
// CHECK: tessera.gelu
// CHECK-SAME: tessera.layout = "col_major"
// CHECK-SAME: tessera.storage_packed = true
// CHECK: tessera.cast
// CHECK-SAME: tessera.layout = "row_major"
// CHECK-SAME: tessera.source_layout = "col_major"
// CHECK-SAME: tessera.storage_packed = true
// CHECK: tessera.reduce
// CHECK-SAME: tessera.layout = "row_major"
// LEGAL-LABEL: func.func @transpose_packed_epilogue_reduce
func.func @transpose_packed_epilogue_reduce(
    %a: tensor<8x4xf32>) -> tensor<4xf32> {
  %seed = "tessera.cast"(%a) {
    tessera.layout = "row_major",
    tessera.storage_container = "int8",
    tessera.storage_pack = {
      container = "int8", factor = 2 : i64, logical = "int4",
      signedness = "signed_twos_complement"
    },
    tessera.storage_packed = true
  } : (tensor<8x4xf32>) -> tensor<8x4xf32>
  %transpose = "tessera.transpose"(%seed) {
    permutation = array<i64: 1, 0>
  } : (tensor<8x4xf32>) -> tensor<4x8xf32>
  %gelu = "tessera.gelu"(%transpose)
      : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %sum = "tessera.reduce"(%gelu) {axis = -1 : i64, kind = "sum"}
      : (tensor<4x8xf32>) -> tensor<4xf32>
  return %sum : tensor<4xf32>
}
