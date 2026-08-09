// RUN: not %tessera_strict_opt -split-input-file %s 2>&1 | FileCheck %s

module {
  func.func @gin_put(%source: memref<8xf32>, %remote: memref<8xf32>) {
    %window = "tessera_collective.window.register"(%remote) {
      window_id = "remote", bytes = 32 : i64, strict_ordering = true
    } : (memref<8xf32>) -> !tessera_collective.window<memref<8xf32>>
    "tessera_collective.put_signal"(%source, %window) {
      count = 8 : i64, dtype = "f32", peer = 1 : i64,
      peer_window_offset = 0 : i64, signal_index = 3 : i64,
      rma_context = 0 : i64
    } : (memref<8xf32>, !tessera_collective.window<memref<8xf32>>) -> ()
    "tessera_collective.wait_signal"() {
      peer = 1 : i64, operation_count = 1 : i64,
      signal_index = 3 : i64, rma_context = 0 : i64
    } : () -> ()
    "tessera_collective.window.deregister"(%window)
      : (!tessera_collective.window<memref<8xf32>>) -> ()
    return
  }
}

// CHECK-DAG: tessera_collective.window.register
// CHECK-DAG: tessera_collective.put_signal
// CHECK-DAG: dtype = "f32"
// CHECK-DAG: peer = 1
// CHECK-DAG: tessera_collective.wait_signal
// CHECK-DAG: tessera_collective.window.deregister

// -----

module {
  func.func @bad_put_dtype(%source: memref<8xf32>, %remote: memref<8xf32>) {
    %window = "tessera_collective.window.register"(%remote) {
      window_id = "remote", bytes = 32 : i64
    } : (memref<8xf32>) -> !tessera_collective.window<memref<8xf32>>
    "tessera_collective.put_signal"(%source, %window) {
      count = 8 : i64, dtype = "f16", peer = 1 : i64,
      peer_window_offset = 0 : i64
    } : (memref<8xf32>, !tessera_collective.window<memref<8xf32>>) -> ()
    return
  }
}

// CHECK-DAG: one-sided dtype disagrees with the source element type

// -----

module {
  func.func @bad_put_extent(%source: memref<8xf32>, %remote: memref<8xf32>) {
    %window = "tessera_collective.window.register"(%remote) {
      window_id = "remote", bytes = 32 : i64
    } : (memref<8xf32>) -> !tessera_collective.window<memref<8xf32>>
    "tessera_collective.put_signal"(%source, %window) {
      count = 8 : i64, dtype = "f32", peer = 1 : i64,
      peer_window_offset = 4 : i64
    } : (memref<8xf32>, !tessera_collective.window<memref<8xf32>>) -> ()
    return
  }
}

// CHECK-DAG: put extent exceeds the peer window

// -----

module {
  func.func @bad_wait() {
    "tessera_collective.wait_signal"() {
      peer = 1 : i64, operation_count = 0 : i64
    } : () -> ()
    return
  }
}

// CHECK-DAG: operation_count must be positive
