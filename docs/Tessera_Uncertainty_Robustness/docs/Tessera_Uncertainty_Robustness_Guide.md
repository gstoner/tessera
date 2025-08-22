# Tessera Uncertainty & Robustness Guide

**Scope.** Make predictive uncertainty a first-class output in Tessera. Covers API shape, training/inference recipes, calibration, conformal intervals, and distributed/deterministic behavior.

> Status: Draft. The examples in `examples/` are runnable PyTorch prototypes that mirror Tessera’s intended API surface.

---

## 1. API Concept

```python
pred = model(x)  # returns a distribution-like object
print(pred.mean, pred.std)
print(pred.epistemic, pred.aleatoric)
pred.quantiles([0.05, 0.95])
pred.interval(coverage=0.9, method="conformal")
```

**Fields**
- `mean`: E[y|x]
- `std`: sqrt Var[y|x]
- `aleatoric`: E_w[ Var(y|x,w) ]
- `epistemic`: Var_w[ E(y|x,w) ]
- `metadata`: distribution type, calibration, sample_count, ensemble_size

---

## 2. GraphIR / ScheduleIR Hooks

- `tessera.graph.uncertainty.capture(samples=S, ensemble=E)` aggregates stochastic passes (dropout/VI) into moments.
- `tessera.schedule.sampling(S, overlap_streams=True)` overlaps S samples across streams; deterministic RNG streams per device.
- Optional `tessera.graph.evidential.dirichlet_head` for classification.

---

## 3. Training Recipes

### 3.1 Regression (heteroscedastic NLL + MC sampling)
- Head outputs `μ(x)` and `log σ²(x)`.
- During inference, run S stochastic forwards to decompose total variance into aleatoric + epistemic.

### 3.2 Classification (MC dropout decomposition)
- S stochastic forwards → class prob tensors `p_t`.
- Total: `H( mean_t p_t )`; Aleatoric: `mean_t H(p_t)`; Epistemic: `H(mean_t p_t) - mean_t H(p_t)`.

### 3.3 Evidential Alternative
- Dirichlet concentration α; mean p = α/Σα; epistemic ~ 1/Σα.

---

## 4. Calibration & Intervals

- **Temperature scaling / isotonic** on held-out calibration split.
- **Conformal prediction (split)** for finite-sample coverage:
  - Maintain normalized residuals; return `[μ̂ ± q̂·σ̂]` at inference.

---

## 5. Robustness Patterns

- Label noise: Laplace / Student-t NLL.
- Adversarial: FGSM/PGD augmentation; gradient penalty; spectral norm.
- Shift: OOD scores, energy regularization; entropy penalty on near-OOD.

---

## 6. Distributed & Determinism

- Per-stream RNG seeds recorded in checkpoints.
- Deterministic reductions for moment aggregation across DP mesh.
- Inference server returns structured payload:
```json
{
  "mean": [...],
  "std": [...],
  "epistemic": [...],
  "aleatoric": [...],
  "quantiles": {"q05": [...], "q95": [...]},
  "coverage": 0.9,
  "samples": 50,
  "ensemble": 1
}
```

---

## 7. Examples

See `examples/`:
- `regression_heteroscedastic_mc_dropout.py` — heteroscedastic NLL + MC dropout with decomposition.
- `classification_mc_dropout.py` — classification entropy/BALD decomposition.
- `conformal_utils.py` — split conformal calibration + intervals.

Run:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
python examples/regression_heteroscedastic_mc_dropout.py
python examples/classification_mc_dropout.py
```
