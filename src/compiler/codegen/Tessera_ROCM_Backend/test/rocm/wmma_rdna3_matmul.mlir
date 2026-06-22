// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1100})' %s | FileCheck %s --check-prefix=WMMA
// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1100},lower-tessera-target-to-rocdl)' %s | FileCheck %s --check-prefix=ROCDL
//
// Strix Halo bring-up (Stage A) — RDNA 3 / 3.5 (gfx1100 on the WSL box,
// gfx1151 native) has no MFMA matrix core; the matmul tile must lower to the
// WMMA matrix op (single 16x16x16 tile, wave32), NOT MFMA. The CDNA path
// (gfx9xx -> tessera_rocm.mfma) is covered by tile_matmul_to_rocm.mlir.

module {
  func.func @wmma_matmul_rdna3(%a: f16, %b: f16) -> f16 {
    %m = "tile.mma"(%a, %b) : (f16, f16) -> f16
    return %m : f16
  }
}

// gfx1100 selects WMMA, not MFMA.
// WMMA: tessera_rocm.wmma
// WMMA-SAME: arch = "gfx1100"
// WMMA-SAME: shape = "m16n16k16"
// WMMA-SAME: source = "tessera.matmul"
// WMMA-NOT: tessera_rocm.mfma

// The WMMA target op lowers to the AMDGCN WMMA artifact marker.
// ROCDL: llvm.func @llvm.amdgcn.wmma.contract
// ROCDL: llvm.call @llvm.amdgcn.wmma.contract
// ROCDL-NOT: llvm.amdgcn.mfma.contract
