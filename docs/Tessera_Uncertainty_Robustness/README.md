# Tessera Uncertainty & Robustness Package

This package accompanies the guide and contains three runnable PyTorch prototypes:
- Heteroscedastic regression + MC dropout with variance decomposition.
- Classification MC dropout with entropy-based aleatoric/epistemic split.
- Conformal prediction utilities (split method).

## Run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
python examples/regression_heteroscedastic_mc_dropout.py
python examples/classification_mc_dropout.py
```
