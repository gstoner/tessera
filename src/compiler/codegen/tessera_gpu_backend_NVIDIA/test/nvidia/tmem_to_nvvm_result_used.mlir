// RUN: not %tnv --lower-tile-to-nvidia='sm=100' --lower-tessera-nvidia-to-nvvm %s 2>&1 | FileCheck %s

// TILE-LATENT-DEFECTS-2026-09-27. Negative twin of tmem_to_nvvm_contract.mlir:
// a TMEM load result consumed outside the contract family cannot be supplied
// by a void artifact marker. The NVVM stage used to `dropAllUses` here and
// leave `func.return` with a null operand; it now fails closed.

module {
  func.func @result_escapes_contracts(%i: index) -> f32 {
    %tmem = tile.tmem.allocate {bytes = 4096 : i64, alignment = 128 : i64}
        : !tile.tmem
    %v = tile.tmem.load %tmem, %i : (!tile.tmem, index) -> f32
    return %v : f32
  }
}

// CHECK: NVIDIA_MARKER_RESULT_USED: 'tessera_nvidia.tmem_load' has no value-producing NVVM lowering, but its result is used by 'func.return'
