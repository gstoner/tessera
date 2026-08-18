// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1100})' %s | FileCheck %s --check-prefix=WMMA
// RUN: not %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1100},lower-tessera-target-to-rocdl)' %s 2>&1 | FileCheck %s --check-prefix=STRICT
//
// Strix Halo bring-up (Stage A) — RDNA 3 / 3.5 (gfx1100 on the WSL box,
// gfx1151 native) has no MFMA matrix core; the matmul tile must lower to the
// WMMA matrix op (single 16x16x16 tile, wave32), NOT MFMA. The CDNA path
// (gfx9xx -> tessera_rocm.mfma) is covered by tile_matmul_to_rocm.mlir.

module {
  func.func @wmma_matmul_rdna3(%a: tensor<16x16xf16>, %b: tensor<16x16xf16>) -> tensor<16x16xf16> {
    %m = "tile.mma"(%a, %b) : (tensor<16x16xf16>, tensor<16x16xf16>) -> tensor<16x16xf16>
    return %m : tensor<16x16xf16>
  }
}

// gfx1100 selects WMMA, not MFMA.
// WMMA: tessera_rocm.wmma
// WMMA-SAME: arch = "gfx1100"
// WMMA-SAME: shape = "m16n16k16"
// WMMA-SAME: source = "tessera.matmul"
// WMMA-NOT: tessera_rocm.mfma

// Scalar WMMA is an inspectable Target-IR contract, not executable hardware
// fragments. Binary lowering must fail rather than replace its value with undef.
// STRICT: executable ROCm matrix lowering requires typed hardware fragment vectors
// STRICT-NOT: .contract
