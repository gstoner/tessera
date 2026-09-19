// The canonical LDS comparison body stores its accumulator at row `2*e + lhi`,
// which is RDNA3's wave32 distribution. gfx12 distributes the same accumulator
// by column, and the `tessera_rocm.wmma` this body emits is the arch-resolving
// Target IR op -- so on gfx12 it would lower to the RDNA4 instruction and then
// scatter the result to RDNA3 rows, running to completion with nothing to
// catch it. Unlike the control-for-WMMA generators, which emit the gfx11 rocdl
// intrinsic and die at instruction selection on gfx12, this body has nothing
// to fail on, so it refuses by name.
//
// This file is the gfx12 refusal; `canonical_lds_arch_guard.mlir` is the
// gfx11 acceptance. Found 2026-09-19 by a structural search for accumulator index
// math carrying no arch branch, on a file a substring audit had cleared.

// RUN: not %trop --allow-unregistered-dialect --generate-wmma-gemm-kernel='canonical-staging=lds' %s 2>&1 | FileCheck %s

module attributes {tessera.arch = "gfx1201"} {
  func.func @canonical_gemm(%a: memref<?xf16>, %b: memref<?xf16>, %d: memref<?xf32>,
                            %M: index, %N: index, %K: index) {
    "tessera_rocm.wmma_gemm"() {
      name = "canonical_gemm", m = 16 : i64, n = 16 : i64, k = 16 : i64,
      mt = 1 : i64, nt = 1 : i64, arch = "gfx1201",
      tessera.canonical_k_loop = true,
      numeric_policy = {storage = "f16", accum = "f32"}
    } : () -> ()
    return
  }
}

// CHECK: error: ROCM_CANONICAL_LDS_ARCH_UNSUPPORTED
