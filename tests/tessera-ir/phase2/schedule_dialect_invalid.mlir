// RUN: not tessera-opt %s 2>&1 | FileCheck %s
//
// E2E-REAL-1: registered Schedule ops run their ODS verifier without the
// unknown-dialect escape hatch.

module {
  schedule.tile {
    source = "tessera.matmul",
    result = "C",
    ordinal = 0 : i64,
    tile_m = 0 : i64
  }
}

// CHECK: error:
// CHECK-SAME: optional tile dimensions must be positive
