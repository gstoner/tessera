# Tessera PyTorch Bridge — Full Demo

This package contains:
- `docs/Tessera_PyTorch_Bridge.md` — full markdown guide.
- `examples/pytorch_bridge/minitransformer.py` — exact example from the doc with a tiny Tessera shim, so it runs even without Tessera installed.

## Run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu  # or a CUDA wheel
python examples/pytorch_bridge/minitransformer.py
```
