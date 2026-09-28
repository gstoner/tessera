// RUN: %tnv --lower-tile-to-nvidia='sm=100' --lower-tessera-nvidia-to-nvvm %s | FileCheck %s

// TILE-LATENT-DEFECTS-2026-09-27. The NVVM stage replaces a contract that has
// no value-producing lowering with a void artifact marker. A value flowing
// only between contracts (the TMEM address into the load/store, the load into
// the store) is erased with them. A value consumed OUTSIDE the contract family
// used to be dropped with `dropAllUses`, leaving its user with a null operand;
// it is now refused (tmem_to_nvvm_result_used.mlir). IR/lowering evidence
// only -- sm_100 is not in the fleet.

module {
  func.func @contract_only_uses(%i: index) {
    %tmem = tile.tmem.allocate {bytes = 4096 : i64, alignment = 128 : i64}
        : !tile.tmem
    %v = tile.tmem.load %tmem, %i : (!tile.tmem, index) -> f32
    tile.tmem.store %v, %tmem [%i] : f32, !tile.tmem, index
    return
  }
}

// CHECK-LABEL: func.func @contract_only_uses
// CHECK: llvm.call @llvm.nvvm.tmem.alloc.contract
// CHECK: llvm.call @llvm.nvvm.tmem.load.contract
// CHECK: llvm.call @llvm.nvvm.tmem.store.contract
// CHECK-NOT: tessera_nvidia.
// CHECK-NOT: tile.tmem
