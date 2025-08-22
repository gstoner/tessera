# Tessera Interpretability Guide

**Goal.** Make interpretability artifacts *first-class* in Tessera so every prediction can return
feature attributions, concept relevance, counterfactuals, and causal structure where feasible.

This guide provides a unified API concept, algorithm choices, IR/runtime hooks, and runnable PyTorch
prototypes that mirror Tessera’s intended behavior.

---

## 1. Unified API Concept

```python
from tessera import explain as X

pred = model(x)  # value or Distribution-like
exp  = X.explain(model, x,
                 methods=["integrated_gradients", "grad_cam", "tcav", "counterfactual", "causal_graph"],
                 target="class:7", steps=64, samples=32)

print(f"Prediction: {getattr(pred,'mean',pred)} ± {getattr(pred,'std','N/A')}")
print("Feature importance shape:", exp.feature_importance.shape)
print("TCAV:", exp.tcav_scores)         # {concept: score}
print("Δx (counterfactual):", exp.counterfactual.dx.norm().item())
print("Causal edges:", (exp.causal_graph.A.abs()>0).sum().item())
```

**Artifact (Explanation) bundle**

- `feature_importance`: IG/Grad-CAM-like heatmap(s)
- `tcav_scores`: dict `{concept → score}` (+ optional confidence / p-values)
- `counterfactual`: `{x_cf, dx, feasibility_meta}`
- `causal_graph`: `{A, effects, confidence}`
- `metadata`: seeds, steps/samples, baselines, determinism flags

---

## 2. Algorithms Provided

### 2.1 Feature Importance
- **Integrated Gradients (IG)** with Riemann/Legendre path and optional SmoothGrad.
- **Grad-CAM/Grad-CAM++** for CNN/ViT-like blocks, using activation maps + gradients. (uses conv/attention maps + dL/dA).
- Token/feature attribution: rollups across channels, timesteps, or heads.

### 2.2 Concept Activation Vectors (TCAV)
- Build **CAVs** at an interior layer from concept vs. random (counter-) examples.
- Score concepts via directional derivatives of the target logit along CAV.

### 2.3 Counterfactual Explanations
- Gradient-based search for minimum Δx achieving a desired target (classification) with proximity
(ℓ1/ℓ2), sparsity, and feasibility constraints (box/monotone/causal). Latent-space versions are possible.
- For images/text, optionally operate in latent space (VAE/encoder) and decode.
- Includes statistical testing over random counter-concepts.

### 2.4 Causal Attribution Graphs
- Lightweight structure learning over features/activations (sparse regression + stability selection),
plus local interventional effect estimates. Produces an adjacency matrix `A` and effect weights on output.
- Interventional effect reports: Δy|do(x_i↑) via local SCM linearization or ablations; aggregates to a causal graph with confidences.


---

## 3. IR & Runtime Hooks

- **Graph IR**
  - `explain.integrated_gradients %x, %x0, steps=N, smooth=S, target=@cls`
  - `explain.grad_cam %act, %dL_dact, pool="mean"`
  - `explain.tcav %H, %concept_set, %random_sets, layer=@Lk`
  - `explain.counterfactual.search %x, target, constraints={...}`
  - `explain.causal.learn %features, method="sparse_reg"`
- **Schedule IR**
  - Batched samples/steps on auxiliary streams; deterministic reductions.
- **Runtime**
  - Activation capture hooks; RNG stream/seed control; mixed-precision policy for explain ops.
- **Inference Server**
  - `?explain=ig,gradcam,...` yields a JSON blob with artifacts and optional overlay images.

---

## 4. Examples (PyTorch Prototypes)

- `integrated_gradients.py` — IG feature attributions on an MLP (tabular).
- `grad_cam.py` — Grad-CAM saliency on a simple CNN.
- `tcav_minimal.py` — minimal TCAV score computation.
- `counterfactual_search.py` — gradient-based CF with proximity and box constraints.
- `causal_graph.py` — toy causal graph recovery from features.

> These mirror Tessera APIs but are framework-only for portability.

---

## 5. Determinism & Performance

- Determinism: fix seeds; capture activation order; use ordered reductions when aggregating IG/TCAV/ensembles across DP ranks.
- Performance: IG/MC sampling parallelized via Schedule IR streams; Grad-CAM uses saved conv/attention maps (no extra forward). Overlap sampling/steps with auxiliary streams; capture activations once (avoid double forward).
- Memory: activation checkpointing for long paths; optional half precision for intermediates (not gradients) with bound on error; keep Grad-CAM intermediates in fp16 if memory-bound.

---

## 6. Running the Examples

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu  # or your CUDA wheel
python examples/integrated_gradients.py
python examples/grad_cam.py
python examples/tcav_minimal.py
python examples/counterfactual_search.py
python examples/causal_graph.py
```

---

## 7. References (selected)

- Sundararajan et al., *Axiomatic Attribution for Deep Networks* (Integrated Gradients).
- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks*.
- Kim et al., *TCAV: Concept Activation Vectors for Explaining Predictions*.
- Wachter et al., *Counterfactual Explanations Without Opening the Black Box*.
- Glymour, Pearl, Jewell, *Causal Inference in Statistics*.
