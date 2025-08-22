# Tessera DNAS Package (v2)

This package adds **GraphIR & ScheduleIR** documentation and a **joint NAS + scheduling** prototype.

## Contents
- `docs/Tessera_DNAS_GraphIR_ScheduleIR.md` — IR-level design (GraphIR & ScheduleIR) with MLIR-style snippets.
- `examples/dnas_graphir_sketch.mlir` — illustrative MLIR-like program.
- `examples/dnas_schedule_autotune.py` — runnable PyTorch prototype (architecture + schedule with a surrogate cost & cache).

## Run the prototype
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu  # or your CUDA wheel
python examples/dnas_schedule_autotune.py
```
You’ll see architecture probabilities and schedule probabilities evolve, and a final **discrete choice** printed for both.
