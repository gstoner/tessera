# Tessera Gemma Port (Starter)

> ⚠️ **Status**: Starter scaffold for porting **Gemma** models to the **Tessera Programming Model**.  
> Implements a clean module layout, reference PyTorch forward passes, and hook points for Tessera kernels / Target IR.  
> No weights included. Use `scripts/convert_hf_gemma_to_tessera.py` to convert HF Gemma checkpoints (after accepting Gemma license).

## What’s here
- `tessera_gemma/model_tessera.py` — Causal LM in PyTorch with hook points for Tessera ops.
- `tessera_gemma/kernels/*` — Attention, RMSNorm, SwiGLU with optional Tessera acceleration.
- `tessera_gemma/ops/*` — RoPE and small utilities.
- `mlir/` — Tessera Target IR stubs for Attention/MLP with comments and lowering hints.
- `docs/Gemma_to_Tessera_Port.md` — Port plan and mapping tables.
- `scripts/convert_hf_gemma_to_tessera.py` — Weights converter (HF → Tessera tensors).
- `tests/` — Shape/KV-cache smoke tests.
- `pyproject.toml` — Editable install as `tessera_gemma`.

## Quickstart
```bash
pip install -e .
python -m tessera_gemma.utils.smoke
pytest -q
```

## Legal
- Source code here is Apache-2.0 (this repo). **Gemma weights** are distributed under Google’s Gemma terms. **Do not** redistribute weights. You must accept and fetch weights through the official gates.
- The converter script expects HuggingFace `google/gemma*` checkpoints you downloaded after accepting terms.
