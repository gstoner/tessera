# Tessera LSL Package

This mini-package demonstrates a **Learning Specification Language (LSL)** for Tessera.

- `tessera/lsl/` — a tiny prototype implementation (`spec.py`, `solver.py`).
- `docs/LSL_Spec.md` — the high-level spec.
- `examples/lsl/vision_rotation_invariance.py` — worked example for image classification with rotation invariance.
- `examples/lsl/llm_uncertainty_energy_cap.py` — worked example for LLM with uncertainty + energy cap.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
# (No external deps required; examples run with the prototype solver)
python examples/lsl/vision_rotation_invariance.py
python examples/lsl/llm_uncertainty_energy_cap.py
```

The prototype prints the constructed spec, the (mock) chosen configuration, and a fake report.
