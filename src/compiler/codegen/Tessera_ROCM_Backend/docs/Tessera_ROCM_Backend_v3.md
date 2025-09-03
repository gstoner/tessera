# Tessera ROCm Backend (v3)

> Merge Marker START
Enhancements:
1) **Async copy → real ops**: lowering uses `llvm.amdgcn.raw.buffer.load.v4f32` + `llvm.amdgcn.ds.write.b128`
   and inserts `llvm.amdgcn.s.barrier` (workgroup barrier) for correctness.
2) **MFMA coverage & tiling**: chooser table varies by `-mcpu` (gfx90a/gfx94/gfx1200). Added GELU epilogue fuse sketch.
3) **Kernel ABI polish**: kernel pass converts `memref<*xT>` arguments to `!llvm.ptr<T, addrspace(1)>` and
   adds `amdgpu-flat-work-group-size` and optional `amdgpu-lds-size`.
4) **Emitter resilience**: prefers `mlir-translate + llc + ld.lld`; falls back to `clang -target amdgcn-amd-amdhsa`.
   Writes `metadata.json` (kernels, wg, LDS).
5) **CK bridge (real path)**: `tessera_ck_bridge` compiles against Composable Kernels if available (stub otherwise).
> Merge Marker END
