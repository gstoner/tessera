// Placeholder for MLIR/LLVM lowering from Tessera Target IR to x86 intrinsics.
// This file is intentionally a skeleton with comments indicating the strategy.
// Integrate by registering these patterns in your compiler pipeline.

/*
Strategy:

1) Detect target features (module attributes, pass options):
   - prefer AMX if available for bf16/int8 mma ops
   - fall back to AVX-512 (BF16 or emulation)

2) Pattern: tile.mma(bf16->fp32)
   - If AMX: emit runtime call to tessera_x86_amx_gemm_bf16(...) for large GEMMs
   - Else if AVX-512 BF16: call tessera_x86_avx512_gemm_bf16(...)
   - For small tiles, inline intrinsics via LLVM IR intrinsics (vd pbf16ps)

3) Pattern: tile.load/store
   - Map to LLVM intrinsics for vector loads/stores (aligned/unaligned) or AMX tile load/store

4) Config ops:
   - Lower config.amx(...) to runtime helper (amx_build_default_bf16_config + amx_load_config)

5) Fused epilogues (bias+relu/gelu):
   - If vectorized: fuse as post-store vector ops (e.g., _mm512_add_ps, _mm512_max_ps, polynomial GELU)
   - With AMX: store TC then fuse with AVX-512 vector ops

This skeleton compiles as part of the library but contains no MLIR deps to keep the sample portable.
*/
