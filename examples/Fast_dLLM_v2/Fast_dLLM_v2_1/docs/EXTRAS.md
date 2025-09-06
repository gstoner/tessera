# Fast‑dLLM v2 extras
This drop adds:
- MLIR pass skeletons for `-tessera-kv-cache-blockify` and `-tessera-parallel-decode-expand`,
- Tile IR ops: `tessera.tile.confidence_stats` and `tessera.tile.prefix_lcp` (TD + C++ stub),
- a tiny microbench (`tools/fast_dllm_microbench`) to replay LLaDA-style steps and print cache reuse stats.
