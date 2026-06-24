// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(rocm-wave-lds-pipeline,rocm-wave-lds-legality,lower-tile-to-rocm{arch=gfx1151})' %s | FileCheck %s --check-prefix=RDNA
// RUN: %trop --allow-unregistered-dialect --pass-pipeline='builtin.module(rocm-wave-lds-pipeline,rocm-wave-lds-legality,lower-tile-to-rocm{arch=gfx942})' %s | FileCheck %s --check-prefix=CDNA

module {
  func.func @matrix_path(%a: f16, %b: f16) -> f16 {
    %m = "tile.mma"(%a, %b) : (f16, f16) -> f16
    return %m : f16
  }
}

// RDNA: tessera_rocm.wmma
// RDNA-SAME: arch = "gfx1151"
// RDNA-NOT: tessera_rocm.mfma

// CDNA: tessera_rocm.mfma
// CDNA-SAME: arch = "gfx942"
// CDNA-NOT: tessera_rocm.wmma
