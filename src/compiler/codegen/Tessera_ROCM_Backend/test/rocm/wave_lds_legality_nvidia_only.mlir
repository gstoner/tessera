// RUN: not %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(rocm-wave-lds-legality)' %s 2>&1 | FileCheck %s

module {
  func.func @nvidia_only_barrier() {
    "test.nvidia_barrier"() {
      tile.barrier = #tile.barrier<kind = "tma", expect = 16384>
    } : () -> ()
    return
  }
}

// CHECK: ROCM_WAVE_LDS_UNSUPPORTED_BARRIER_KIND
