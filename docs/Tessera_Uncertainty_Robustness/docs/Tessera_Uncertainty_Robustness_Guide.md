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
Data Model 
**Fields**
- `mean`: E[y|x]
- `std`: sqrt Var[y|x]
- `aleatoric`: E_w[ Var(y|x,w) ]
- `epistemic`: Var_w[ E(y|x,w) ]
- `metadata`: distribution type, calibration, sample_count, ensemble_size

Decomposition:  Var[y|x] = E_w[var(y|x,w)] + var_w[E(y|x,w)]
(aleatoric)                     ( epistemic )
---



## 2. GraphIR / ScheduleIR Hooks
How Tessera would surface it
	•	Graph IR adds stochastic nodes (rand_key, dropout(mode="sampling"), bayes.sample(w)), and a uncertainty.capture op that aggregates multiple stochastic passes into {mean, std, aleatoric, epistemic} tensors with well-defined reduction order (deterministic).
	•	Schedule IR can request S Monte-Carlo samples and overlaps them across streams; also supports ensembles (E members × S samples).
	•	Runtime maintains RNG streams per device/stream; seeds are recorded in checkpoints so results are reproducible.
	•	Inference server returns a structured JSON payload (means, intervals, decomposition, calibration metrics), with an option for conformal guarantees.

- `tessera.graph.uncertainty.capture(samples=S, ensemble=E)` aggregates stochastic passes (dropout/VI) into moments.
- `tessera.schedule.sampling(S, overlap_streams=True)` overlaps S samples across streams; deterministic RNG streams per device.
- Optional `tessera.graph.evidential.dirichlet_head` for classification.

---
3) Practical recipes

A Regression (heteroscedastic + MC sampling)

	-	Model outputs μ(x) and log σ²(x) (data noise).
	-	Add MC Dropout or SWAG/VI to capture epistemic.
	-	Aggregate across S samples:
	-	μ̂ = mean_t μ_t
	-	aleatoric = mean_t σ_t²
	-	epistemic = var_t μ_t
	-	std = sqrt(aleatoric + epistemic)

B Classification (MC sampling)

	-	Run S stochastic forwards → class prob vectors p_t.
	-	Total uncertainty: H( mean_t p_t )
	-	Aleatoric: mean_t H(p_t)
	-	Epistemic (BALD): H(mean_t p_t) - mean_t H(p_t)

C Evidential (Dirichlet) alternative

	-	Head outputs Dirichlet concentration α; mean probs p = α / Σα.
	-	Use evidential loss (e.g., NLL + evidence regularizer) and derive uncertainty from α magnitude (higher Σα ⇒ lower epistemic).

D Conformal intervals (finite-sample coverage)
	-	Maintain calibration residuals; at inference compute quantile q̂ and return [μ̂−q̂, μ̂+q̂]. Works orthogonally to the above.

E Robustness training hooks
	-	Label-noise & heavy-tail losses: Gaussian NLL, Laplace NLL, Student-t NLL.
	-	Adversarial/robust: PGD/FGSM augmentations; gradient penalty; spectral norm or Lipschitz control.
	-	Distribution shift: energy/OOD scores, temperature scaling, entropy regularization.

4) Minimal working prototype (PyTorch)

4.1 Regression: heteroscedastic head + MC Dropout
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

## 5. Calibration & Intervals

- **Temperature scaling / isotonic** on held-out calibration split.
- **Conformal prediction (split)** for finite-sample coverage:
  - Maintain normalized residuals; return `[μ̂ ± q̂·σ̂]` at inference.

---

## 6. Robustness Patterns

- Label noise: Laplace / Student-t NLL.
- Adversarial: FGSM/PGD augmentation; gradient penalty; spectral norm.
- Shift: OOD scores, energy regularization; entropy penalty on near-OOD.

---

## 7. Distributed & Determinism

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

## 8. Pytorch examples to use as reference 

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
## 9. Minimal working prototype (PyTorch)

### 9.1 Regression: heteroscedastic head + MC Dropout
```python
import torch, torch.nn as nn, torch.nn.functional as F

class Pred:
    def __init__(self, mean, std, epistemic=None, aleatoric=None):
        self.mean = mean
        self.std = std
        self.epistemic = epistemic
        self.aleatoric = aleatoric
    def quantiles(self, qs):
        # assuming Gaussian for demo
        from math import sqrt
        q = torch.tensor(qs, device=self.mean.device)  # z-score approx via erfinv if needed
        # quick-and-dirty normal quantiles via inverse error function:
        z = torch.sqrt(torch.tensor(2.0, device=self.mean.device)) * torch.erfinv(2*q-1)
        return [(self.mean + z_i*self.std) for z_i in z]

class HetReg(nn.Module):
    def __init__(self, d, hidden=128, pdrop=0.1):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(),
            nn.Dropout(pdrop),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(pdrop)
        )
        self.mu = nn.Linear(hidden, 1)
        self.logvar = nn.Linear(hidden, 1)   # predicts log σ²(x)
    def forward(self, x):
        h = self.f(x)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-10, 5)
        return mu, logvar

def gaussian_nll(mu, logvar, y):
    inv = torch.exp(-logvar)
    return 0.5*(inv*(y-mu)**2 + logvar).mean()

@torch.no_grad()
def predict_with_uncertainty(model, x, samples=20, train_mode=True):
    # Enable dropout sampling
    was_training = model.training
    if train_mode: model.train()
    else: model.eval()

    mus, sig2s = [], []
    for _ in range(samples):
        mu, logvar = model(x)
        mus.append(mu)
        sig2s.append(torch.exp(logvar))
    mus = torch.stack(mus, 0)[...,0]     # [S,B]
    sig2s = torch.stack(sig2s, 0)[...,0] # [S,B]

    mu_hat = mus.mean(0)                 # [B]
    aleatoric = sig2s.mean(0)            # E[σ²]
    epistemic = mus.var(0, unbiased=False)  # Var[μ]
    std = (aleatoric + epistemic).sqrt()
    if was_training: model.train()
    return Pred(mu_hat, std, epistemic, aleatoric)

# --- tiny demo ---
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(512, 8); y = (x[:,0]*2 + 0.3*torch.randn_like(x[:,0])).unsqueeze(-1)
    m = HetReg(d=8, pdrop=0.2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(400):
        mu, logv = m(x)
        loss = gaussian_nll(mu, logv, y)
        opt.zero_grad(); loss.backward(); opt.step()
    pred = predict_with_uncertainty(m, x[:4], samples=50)
    print("mean:", pred.mean.flatten().tolist())
    print("std:", pred.std.flatten().tolist())
    print("epi:", pred.epistemic.flatten().tolist())
    print("ale:", pred.aleatoric.flatten().tolist())
```
### 9.2 Classification: MC dropout decomposition
```python 
import torch, torch.nn as nn, torch.nn.functional as F

class Clf(nn.Module):
    def __init__(self, d, k, pdrop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(), nn.Dropout(pdrop),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(pdrop),
            nn.Linear(128, k)
        )
    def forward(self, x): return self.net(x)

@torch.no_grad()
def classify_with_uncertainty(model, x, samples=20):
    was_training = model.training
    model.train()  # enable dropout sampling
    probs = []
    for _ in range(samples):
        logits = model(x)
        probs.append(F.softmax(logits, dim=-1))
    P = torch.stack(probs, 0)                 # [S,B,K]
    p_mean = P.mean(0)                        # [B,K]
    # Total entropy H(E[p])
    total_unc = -(p_mean * (p_mean.clamp_min(1e-8)).log()).sum(-1)
    # Aleatoric: E[H(p)]
    ale = (-(P * P.clamp_min(1e-8).log()).sum(-1)).mean(0)
    # Epistemic (BALD): H(E[p]) - E[H(p)]
    epi = total_unc - ale
    if not was_training: model.eval()
    return p_mean, epi, ale

# quick demo
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(256, 16); y = (x[:,0] > 0).long()
    m = Clf(16, 2, 0.2)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    for _ in range(300):
        m.train(); logits = m(x); loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
    p, epi, ale = classify_with_uncertainty(m, x[:8], samples=50)
    print("p:", p.tolist()); print("epi:", epi.tolist()); print("ale:", ale.tolist())
```
### 9.3 Conformal prediction (regression, split method)
```python
@torch.no_grad()
def conformal_calibrate(model, x_cal, y_cal, samples=30):
    # Use predictive std as a scale; compute absolute residuals normalized by std
    pred = predict_with_uncertainty(model, x_cal, samples=samples)
    r = (y_cal.squeeze(-1) - pred.mean).abs() / pred.std.clamp_min(1e-6)
    return r.sort().values

@torch.no_grad()
def conformal_interval(model, x, cal_scores, alpha=0.1, samples=30):
    q_idx = int((1 - alpha) * (len(cal_scores) + 1)) - 1
    q = cal_scores[q_idx.clamp(0, len(cal_scores)-1)]
    pred = predict_with_uncertainty(model, x, samples=samples)
    lo = pred.mean - q*pred.std
    hi = pred.mean + q*pred.std
    return lo, hi
```
## 10 Training & calibration tips

	-	Losses
	-	Regression: Gaussian/Laplace/Student-t NLL (pick by noise profile).
	-	Classification: cross-entropy; for evidential, add evidence regularizer.
	-	Calibration
	-	Temperature scaling or isotonic on a held-out set; report ECE, NLL, Brier.
	-	Use conformal intervals if you need coverage guarantees.
	-	Efficiency
	-	Share activations across samples where possible (S small, ensemble E small); cache dropout masks for reproducibility; overlap S on streams.
	-	Distributed
	-	Aggregate moments with deterministic order; store RNG seeds per rank in checkpoints.

## 11) How this plugs into Tessera docs/code you already have

	-	Add a “Uncertainty & Robustness Guide” to the docs set (I can spin up a markdown + example zip next).
	-	Extend the inference server schema to return {mean,std,epistemic,aleatoric,quantiles,conformal}.
	-	Add TSOL operators:
	-	tsol.uncertainty.capture(samples=S, ensemble=E)
	-	tsol.calibrate.temperature() / tsol.calibrate.isotonic()
	-	tsol.conformal.split(cal_set) for regression PIs.
	-	Compiler/runtime: RNG stream plumbing, deterministic reductions, and stream-overlapped sampling knobs in Schedule IR.