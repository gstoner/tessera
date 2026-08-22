// RUN: not %tnv %s 2>&1 | FileCheck %s
//
// Target IR must not accept arbitrary SSA types. Async completion tokens belong
// to the Tile synchronization layer and cannot be smuggled into a physical
// NVIDIA MMA contract as an untyped operand.

func.func @reject_sync_token(%token: !tile.async_token) {
  tessera_nvidia.mma_sync %token
      {arch = "sm_120", shape = "m16n8k16", dtype_ab = "f16", dtype_c = "f32"}
      : (!tile.async_token) -> ()
  return
}

// CHECK: error: 'tessera_nvidia.mma_sync' op operand #0 must be variadic of a NVIDIA data buffer, scalar/vector fragment, or LLVM ABI value
