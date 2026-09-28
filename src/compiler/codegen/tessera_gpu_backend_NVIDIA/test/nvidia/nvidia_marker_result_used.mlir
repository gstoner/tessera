// RUN: not %tnv --lower-tessera-nvidia-to-nvvm %s 2>&1 | FileCheck %s

// TILE-LATENT-DEFECTS-2026-09-27. Target-IR-level twin of
// tmem_to_nvvm_result_used.mlir, isolating the NVVM stage: a contract with no
// value-producing NVVM lowering becomes a void artifact marker, so a result
// consumed outside the contract family cannot be supplied. The pass used to
// `dropAllUses` and erase, leaving `func.return` with a null operand. It now
// fails closed with a diagnostic naming the contract and the user.

module {
  func.func @marker_result_escapes(%addr: i32, %i: i64) -> f32 {
    %v = tessera_nvidia.tmem_load %addr, %i {arch = "sm_100a"} : (i32, i64) -> f32
    return %v : f32
  }
}

// CHECK: NVIDIA_MARKER_RESULT_USED: 'tessera_nvidia.tmem_load' has no value-producing NVVM lowering, but its result is used by 'func.return'
