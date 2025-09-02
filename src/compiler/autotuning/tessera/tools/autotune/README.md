
# Tessera Autotune Tools

This directory contains a production-ready skeleton for integrating autotuning into the Tessera compiler flow.

## Highlights
- **MLIR**: `mlir/gemm_meta_schedule.mlir` (transform dialect sketch) + `include/tessera/TunableAttrs.td`
- **Python tuner**: SQLite cache + schedule keys; Grid/Random/**Hyperband**
- **Docs**: split architecture doc with merge markers
- **Examples/Configs**: synthetic GEMM with Hyperband

## Quickstart
```bash
# Run Hyperband
python -m tessera.tools.autotune.tessera_autotuner.cli \
  --config tessera/tools/autotune/configs/matmul_sm90.json \
  --algo hyperband -o runs/hb
```

## Merge the docs
Markers are present so you can combine Parts 1 & 2 into `Tessera_Autotune_Architecture.md`.
