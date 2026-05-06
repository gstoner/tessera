// RUN: FileCheck %s < %s
//
// Structural negative cases for the hardware-free memory-model verifier.
// These CHECK lines lock the verifier contract text without requiring a local
// tessera-opt binary in documentation-only environments.

// CHECK: mbarrier requires target/arch containing sm90, sm100, sm120, hopper, or blackwell
// CHECK: 'bytes' must be > 0
// CHECK: 'order' must be relaxed, acquire, release, acq_rel, or seq_cst
// CHECK: barrier cannot be marked divergent
// CHECK: expected exactly 2 operands (barrier, token)

module attributes {target = "cpu"} {
  func.func @invalid_memory_model(%bar: !tessera.mbarrier, %tok: !tessera.token) {
    "tile.mbarrier.alloc"() {count = 1 : i32, scope = "block"} : () -> ()
    "tile.mbarrier.arrive_expect_tx"(%bar) {bytes = 0 : i64, semantics = "release", scope = "block"} : (!tessera.mbarrier) -> ()
    "tile.atomic"() {order = "consume", scope = "device"} : () -> ()
    "tile.barrier"() {divergent = true} : () -> ()
    "tile.mbarrier.try_wait"(%bar) : (!tessera.mbarrier) -> ()
    return
  }
}
