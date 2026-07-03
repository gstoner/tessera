// RUN: not %tnv --allow-unregistered-dialect --lower-tile-to-nvidia='sm=90' %s 2>&1 | FileCheck %s
// RUN: not %tnv --allow-unregistered-dialect --lower-tile-to-nvidia='sm=120' %s 2>&1 | FileCheck %s --check-prefix=CONSUMER

module {
  func.func @kernel() {
    "tile.tmem.load"() : () -> ()
    return
  }
}

// TMEM is datacenter-Blackwell-only (sm_100a). Rejected below SM100 (Hopper
// sm_90) AND on consumer Blackwell (sm_120, which has no TMEM — it is not a
// superset of datacenter sm_100).
// CHECK: NVIDIA TMEM lowering requires datacenter Blackwell SM100
// CONSUMER: NVIDIA TMEM lowering requires datacenter Blackwell SM100
