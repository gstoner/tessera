// The canonical LDS comparison body stores its accumulator at row `2*e + lhi`,
// which is RDNA3's wave32 distribution. gfx12 distributes the same accumulator
// by column, and the op this body emits is `tessera_rocm.wmma` -- the
// ARCH-RESOLVING Target IR op -- so on gfx12 it would lower to the RDNA4
// instruction and then write the result to RDNA3 rows: the kernel runs to
// completion and a few tiles are silently wrong.
//
// That is what separates it from the control-for-WMMA generators, which
// hardcode the same RDNA3 formula but emit the gfx11 rocdl intrinsic directly
// and therefore die at instruction selection on gfx12. This body had nothing
// to fail on, so it refuses by name.
//
// The body is reached only by the canonical M/N/K scf.for MATCHER, never by an
// attribute, so these fixtures drive the real tiling pipeline. An earlier cut
// of them hand-wrote a `tessera_rocm.wmma_gemm` op and silently exercised the
// general body instead, which is why both passed while proving nothing.
//
// gfx11 acceptance; the refusal is `canonical_lds_arch_refused.mlir`.

// RUN: %trop --tessera-tiling --tessera-tile-ir-lowering --rocm-wave-lds-pipeline --rocm-wave-lds-legality --generate-wmma-gemm-kernel='canonical-staging=lds' %s | FileCheck %s

module attributes {tessera.arch = "gfx1151"} {
  func.func @canonical_gemm(%a: tensor<31x23xf16>, %b: tensor<23x47xf16>) -> tensor<31x47xf32> {
    %0 = "tessera.matmul"(%a, %b) : (tensor<31x23xf16>, tensor<23x47xf16>) -> tensor<31x47xf32>
    return %0 : tensor<31x47xf32>
  }
}

// CHECK: gpu.func @canonical_gemm
// CHECK-SAME: workgroup(
// CHECK-SAME: memref<256xf16, #gpu.address_space<workgroup>>
// CHECK-SAME: tessera.rocm.physical_staging = "lds"
