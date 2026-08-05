// RUN: tessera-opt %s | FileCheck %s
//
// E2E-REAL-1: Schedule IR is a production-registered dialect.  This fixture
// deliberately omits --allow-unregistered-dialect.

module {
  schedule.tile {
    source = "tessera.matmul",
    result = "C",
    ordinal = 0 : i64,
    tile_m = 64 : i64,
    tile_n = 32 : i64,
    tile_k = 16 : i64
  }
  schedule.artifact {
    hash = "0123456789abcdef",
    arch = "x86",
    shape_key = "M=64;N=32;K=16;dtype=f32"
  }
}

// CHECK: schedule.tile
// CHECK-SAME: ordinal = 0
// CHECK-SAME: result = "C"
// CHECK-SAME: source = "tessera.matmul"
// CHECK: schedule.artifact
// CHECK-SAME: arch = "x86"
// CHECK-SAME: hash = "0123456789abcdef"
