# Tessera Flash Attention — Demo Skeleton

This example mirrors your outline and adds:
- A **reference SDPA** (scaled dot-product attention) in PyTorch for validation.
- A **Tessera path** (`tsr.nn.flash_attention`) if available.
- CLI flags for sizes, dtype, device, and causal.
- Optional **IR dump hooks** (`--dump-ir`) that try common debug entry points
  to save `graph_ir.mlir`, `schedule_ir.mlir`, `tile_ir.mlir`, `target_ir.mlir` into `artifacts/`.
- Timing and basic correctness checks (when both paths are available).

## Quick start

```bash
python examples/flash_attention_demo.py   --batch 4 --heads 12 --seq 2048 --dim 64   --dtype bf16 --device auto --causal --dump-ir
```

If Tessera isn't installed yet, the script still runs the **reference** path and
writes placeholder IR files when `--dump-ir` is set.

Artifacts:
- `artifacts/{graph_ir,schedule_ir,tile_ir,target_ir}.mlir`
- `artifacts/compilation_summary.json`
- `artifacts/timings.json`

> Adjust `DUMP_SPEC` in the script to match your local Tessera debug API names.