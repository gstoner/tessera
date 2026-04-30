# Differentiable NAS Compiler Examples

These examples show how Tessera represents differentiable architecture search
at the compiler boundary.

## Files

- `dnas_graphir_sketch.mlir` - Graph IR sketch for architecture parameters,
  relaxed gates, and weighted candidate ops.
- `dnas_schedule_autotune.py` - runnable Python schedule-search and hardware
  cost-model example using `tessera.arch`.

## Run

From the repo root:

```bash
PYTHONPATH=python python3 examples/compiler/dnas/dnas_schedule_autotune.py
```

This keeps model architecture choices in Graph IR and schedule knobs in
Schedule IR, then freezes both by argmax after search.
