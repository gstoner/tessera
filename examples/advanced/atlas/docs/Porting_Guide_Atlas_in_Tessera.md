<!-- MERGE_START: ATLAS_PORT_GUIDE -->
# Porting Guide: Implementing Atlas in Tessera (v0.1) — 2025-09-17

## Dialect & Ops
We introduce `atlas` dialect with:
- `atlas.memory.create/reset/read/update`
- `atlas.optimizer.set(name="muon"| "gd" | "mom", lr=..., beta1=..., beta2=...)`
- `atlas.feature.map(kind="poly"| "exp"| "rff", degree=...)`

## Pass Pipeline (alias: `-tessera-atlas`)
1. `-tessera-atlas-window-plan` — choose window `W`/stride per layer based on HBM/L2 and KV paging.
2. `-tessera-atlas-featuremap-lower` — lower feature maps to Tile IR/Target IR.
3. `-tessera-atlas-memory-lower` — fuse update/read; materialize optimizer math (Muon) and staging.

## Runtime/Perf Hints
- Page memory tiles to KV-cache style buffers; use token-level streams.
- Overlap `feature.map` for token t+Δ with `memory.update` of t (double-buffer).
- Prefer tensorcore/MFMA paths for poly features (matmul expansions).

## Numerics
- Muon-style update may benefit from FP32 accum + BF16/FP16 storage.
- Clamp/normalize window statistics to avoid drift; optional EMA for gates.

## Testing
- FileCheck tests for pass effects.
- Microbench: sweep W, degree, d_model and compare NLL vs baseline.

<!-- MERGE_END: ATLAS_PORT_GUIDE -->