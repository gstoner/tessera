// RUN: tessera-opt --tessera-resilience-restart="restart-policy=last max-restarts=3" %s | FileCheck %s

// Test: ResilienceRestartPass wraps functions in resilience_region and
// inserts tessera_sr.save / tessera_sr.restore hook annotations.

module {

  // CHECK-LABEL: func.func @forward
  // CHECK: tessera_sr.resilience_region
  // CHECK: tessera_sr.restart_policy
  func.func @forward(%x: memref<128x256xbf16>) -> memref<128x256xbf16> {
    %out = memref.alloc() : memref<128x256xbf16>

    // CHECK: tessera_sr.restore_hook
    "tessera.matmul"(%x, %out) {} :
        (memref<128x256xbf16>, memref<128x256xbf16>) -> ()

    // CHECK: tessera_sr.save_hook
    // CHECK-SAME: tsrCheckpointSave
    return %out : memref<128x256xbf16>
  }

  // CHECK-LABEL: func.func @backward
  // CHECK: tessera_sr.resilience_region
  func.func @backward(%grad: memref<128x256xbf16>) -> memref<128x256xbf16> {
    %out = memref.alloc() : memref<128x256xbf16>
    "tessera.matmul"(%grad, %out) {} :
        (memref<128x256xbf16>, memref<128x256xbf16>) -> ()
    return %out : memref<128x256xbf16>
  }
}
