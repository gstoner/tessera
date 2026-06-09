// RUN: tessera-opt --tessera-resilience-restart="restart-policy=last max-restarts=3" --allow-unregistered-dialect %s | FileCheck %s

// Test: ResilienceRestartPass marks the module + functions with resilience
// metadata and inserts tessera_sr.restore_hook / save_hook ABI annotations.
//
// 2026-06: un-XFAIL'd.  Value-semantics tensor matmul (MLIR-22 verifier) +
// --allow-unregistered-dialect for the tessera_sr.* annotation markers.

// CHECK: tessera_sr.restart_policy
module {

  // CHECK-LABEL: func.func @forward
  // CHECK-SAME: tessera_sr.resilience_region
  func.func @forward(%x: tensor<128x256xbf16>, %w: tensor<256x256xbf16>)
      -> tensor<128x256xbf16> {

    // CHECK: tessera.matmul
    // CHECK-SAME: tessera_sr.restore_hook
    %out = "tessera.matmul"(%x, %w) {} :
        (tensor<128x256xbf16>, tensor<256x256xbf16>) -> tensor<128x256xbf16>

    // CHECK: return
    // CHECK-SAME: tessera_sr.abi_call = "tsrCheckpointSave"
    // CHECK-SAME: tessera_sr.save_hook
    return %out : tensor<128x256xbf16>
  }

  // CHECK-LABEL: func.func @backward
  // CHECK-SAME: tessera_sr.resilience_region
  func.func @backward(%grad: tensor<128x256xbf16>, %w: tensor<256x256xbf16>)
      -> tensor<128x256xbf16> {
    %out = "tessera.matmul"(%grad, %w) {} :
        (tensor<128x256xbf16>, tensor<256x256xbf16>) -> tensor<128x256xbf16>
    return %out : tensor<128x256xbf16>
  }
}
