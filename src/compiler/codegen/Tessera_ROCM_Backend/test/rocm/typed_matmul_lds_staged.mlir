// RUN: %trop --allow-unregistered-dialect --generate-wmma-gemm-kernel='via-tile=true canonical-staging=lds lds-waves-m=2 lds-waves-n=2' %s | FileCheck %s --check-prefix=GEN
// RUN: %trop --allow-unregistered-dialect --generate-wmma-gemm-kernel='via-tile=true canonical-staging=lds lds-waves-m=2 lds-waves-n=2' --lower-tile-to-rocm='arch=gfx1201' %s | FileCheck %s --check-prefixes=LOW,RDNA4
// RUN: %trop --allow-unregistered-dialect --generate-wmma-gemm-kernel='via-tile=true canonical-staging=lds lds-waves-m=2 lds-waves-n=2' --lower-tile-to-rocm='arch=gfx1151' %s | FileCheck %s --check-prefixes=LOW,RDNA3
//
// The LDS-staged multi-wave typed body (typed-route gap, 2026-09-18). WM x WN
// waves per workgroup each own the Schedule's macro tile, so the block tile is
// (WM*macro_m) x (WN*macro_n) and the block carries 32*WM*WN threads. Each
// 16-wide K slab is staged cooperatively with B TRANSPOSED into LDS, so both
// fragment packs are contiguous vector loads from shared memory rather than
// the strided gather a row-major B forces in global memory. The fragment ops
// are the register body's, so the per-chip layout comes from the same
// materializer: this body is not pinned to either chip and must NOT claim
// gfx1151's 2x4 register-panel contract.

module {
  func.func @gemm(%a: !llvm.ptr, %b: !llvm.ptr, %d: !llvm.ptr, %m: i64, %n: i64, %k: i64) {
    tile.matmul_kernel %a, %b, %d, %m, %n, %k {
      mma = #tile.mma_desc<family = "wmma", m = 16, n = 16, k = 16, a = "f16", b = "f16", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "f32">,
      warps = 1 : i64, staging = "global",
      tessera.macro_tile_m = 32 : i64, tessera.macro_tile_n = 64 : i64
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    return
  }
}

// 2x2 waves over a 32x64 panel: a 64x128 block tile, 128 threads, and one
// 16-wide K slab of each operand in LDS.
//
// The rows are PADDED (ROCM-LDS-BANKPAD-1, default `lds-pad-dwords=1`), so the
// stride is 18 f16 rather than 16: an unpadded 16-element row is 8 dwords and
// `gcd(8, 32) = 8`, which puts sixteen lanes on four banks -- a 4-way conflict
// on every fragment read. One dword of padding makes the stride 9 dwords, and
// `gcd(9, 32) = 1` makes `(9L) mod 32` a bijection over the lanes.
//
// So 64*18 + 128*18 halves = 6912 bytes, not 64*16 + 128*16 = 6144. Measured
// worth +10-12% on the LDS body; if these numbers ever go back to 1024/2048/6144
// the padding has been lost, which is silent at runtime and only shows as a
// throughput regression.
// GEN: gpu.func @gemm
// GEN-SAME: workgroup(%{{.*}}: memref<1152xf16, #gpu.address_space<workgroup>>, %{{.*}}: memref<2304xf16, #gpu.address_space<workgroup>>)
// GEN-SAME: known_block_size = array<i32: 128, 1, 1>
// GEN-DAG: tessera.rocm.lds_bytes = 6912
// GEN-DAG: tessera.rocm.lds_pad_dwords = 1
// GEN-DAG: tessera.rocm.lds_waves = array<i64: 2, 2>
// An LDS body never claims the gfx11 register-panel contract.
// GEN-NOT: tessera.rocm.typed_gfx11_gemm_contract
// GEN: gpu.barrier
// GEN: tile.fragment_pack
// GEN: tile.mma

// Both chips lower it, each with its own fragment family, and every pack is a
// contiguous vector load from the workgroup buffer.
// LOW: gpu.barrier
// LOW: vector.load %{{.*}} : memref<{{[0-9]+}}xf16, #gpu.address_space<workgroup>>
// LOW: tessera_rocm.wmma
// RDNA4-SAME: fragment_family = "rdna4_wmma"
// RDNA3-SAME: fragment_family = "rdna3_wmma"
