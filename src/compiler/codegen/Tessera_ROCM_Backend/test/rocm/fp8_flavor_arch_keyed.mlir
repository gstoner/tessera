// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx942})' %s | FileCheck %s --check-prefix=FNUZ
// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx950})' %s | FileCheck %s --check-prefix=OCP
//
// A6 / B4 — the SAME canonical fp8 dtype lowers to a different flavor per arch:
// FNUZ on CDNA 3 (gfx942 = E4M3FNUZ), OCP-plain on CDNA 4 (gfx950 = E4M3).
// The flavor is derived in TileToROCM.cpp from the arch-keyed FP8 semantics
// table (mirror of tessera.compiler.rocm_target._FP8_SEMANTICS).

module {
  func.func @fp8_mma(%a: f8E4M3FN, %b: f8E4M3FN) -> f8E4M3FN {
    %m = "tile.mma"(%a, %b) : (f8E4M3FN, f8E4M3FN) -> f8E4M3FN
    return %m : f8E4M3FN
  }
}

// FNUZ: tessera_rocm.mfma
// FNUZ-SAME: arch = "gfx942"
// FNUZ-SAME: fp8_flavor = "e4m3fnuz"

// OCP: tessera_rocm.mfma
// OCP-SAME: arch = "gfx950"
// OCP-SAME: fp8_flavor = "e4m3"
