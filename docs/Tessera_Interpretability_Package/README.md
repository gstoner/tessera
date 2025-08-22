# Tessera Interpretability Package

This package contains a guide and five runnable PyTorch examples that mirror Tessera’s planned
interpretability APIs (IG, Grad-CAM, TCAV, Counterfactuals, Causal Graphs).

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
python examples/integrated_gradients.py
python examples/grad_cam.py
python examples/tcav_minimal.py
python examples/counterfactual_search.py
python examples/causal_graph.py
```
