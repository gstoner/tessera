
# Tessera Rubin CPX Compiler Support (v1.1)

**What’s new (relative to v1):**
1. **NVFP4 legalization marks**: attach `tessera.nvfp4.*` attrs to matmul/attention ops (tile/accum).
2. **Policy-controlled KV lowering**: routes to `@tessera_kv_send_pcie_cx9` or `@tessera_kv_send_nvlink` based on op attr.
3. **Async bridge tokens**: `kv.export` now returns `!async.token`; `kv.import` awaits the token.
4. **Video fuse**: recognizes `video.decode → patchify → tokenizer → prefill_fused` and outlines into `tessera.target.cpx.video.ingest_fused`.

## Build
```bash
cmake -S . -B build -G Ninja -DMLIR_DIR=<path-to-mlir-cmake>
cmake --build build
```

## Smoke tests
```bash
tessera-cpx-opt test/partition/longcontext_split_async.mlir -tessera-partition-longcontext | FileCheck test/partition/longcontext_split_async.mlir
tessera-cpx-opt test/kv/lower_kv_policy.mlir -tessera-lower-kv-transport | FileCheck test/kv/lower_kv_policy.mlir
tessera-cpx-opt test/vec/nvfp4_marks.mlir -tessera-vectorize-nvfp4 | FileCheck test/vec/nvfp4_marks.mlir
tessera-cpx-opt test/video/fuse_ingest_chain.mlir -tessera-fuse-video-ingest | FileCheck test/video/fuse_ingest_chain.mlir
```
