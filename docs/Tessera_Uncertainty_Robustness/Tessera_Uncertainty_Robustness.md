# Tessera Uncertainty & Robustness Guide

This guide introduces **first-class support for uncertainty and robustness** in the Tessera programming model.

---

## 1. Prediction as a First-Class Value

Tessera models return structured prediction objects with uncertainty decomposition:

```python
pred = model(x)
print(f"Prediction: {pred.mean} ± {pred.std}")
print(f"Epistemic uncertainty: {pred.epistemic}")
print(f"Aleatoric uncertainty: {pred.aleatoric}")
```

### Key attributes
- **mean**: point estimate
- **std**: total predictive stddev
- **epistemic**: model uncertainty
- **aleatoric**: data noise uncertainty

---

## 2. API Extensions

- `quantiles([0.05, 0.95])`
- `interval(coverage=0.9, method="conformal")`
- `entropy()`, `mutual_information()`

---

## 3. Recipes

### Regression (MC Dropout + heteroscedastic head)
- Train with Gaussian NLL
- Aggregate S samples for aleatoric + epistemic decomposition

### Classification (MC Dropout)
- Run S stochastic forwards
- Use **total entropy**, **expected entropy**, and **BALD** for uncertainty split

### Evidential Alternative
- Dirichlet concentration α as output
- Aleatoric vs. epistemic via Σα

### Conformal Intervals
- Calibration set for finite-sample guarantees

---

## 4. Code Examples

See Python reference implementations in `examples/`:

- `regression_uncertainty.py` — heteroscedastic regression with MC dropout
- `classification_uncertainty.py` — classification uncertainty decomposition
- `conformal_prediction.py` — conformal intervals

---

## 5. Training & Calibration

- Regression: Gaussian / Laplace / Student-t NLL
- Classification: cross-entropy + calibration
- Calibration methods: temperature scaling, isotonic regression, conformal

---

## 6. Integration in Tessera

- **IR support**: stochastic ops + uncertainty capture
- **Schedule IR**: multi-sample aggregation with overlapped streams
- **Runtime**: RNG streams, reproducible seeds, deterministic reductions
- **Inference server**: structured JSON output

---

## 7. Best Practices

- Choose the uncertainty method suited to the task
- Always calibrate (ECE, NLL, Brier score)
- For distributed training: aggregate uncertainty with deterministic reduction

---
