# Tessera Interpretability Guide

This guide introduces Tessera's **first-class interpretability artifacts**:

- **Feature Importance Maps** (Integrated Gradients, Grad-CAM)
- **Concept Activation Vectors (TCAV)**
- **Counterfactual Explanations**
- **Causal Attribution Graphs**

## Unified API Surface

```python
from tessera import explain as X

pred = model(x)
exp  = X.explain(model, x, methods=["integrated_gradients","grad_cam","tcav","counterfactual","causal_graph"],
                 target="class:7", samples=64, steps=64)

print(f"Prediction: {pred.mean} ± {pred.std}")
print(f"Importance map:", exp.feature_importance.shape)
print(f"TCAV scores:", exp.tcav_scores)
print(f"Counterfactual Δx:", exp.counterfactual.dx)
print(f"Causal graph A_ij:", exp.causal_graph.A)
```

Each explanation returns a structured `Explanation` bundle with feature maps, scores, counterfactuals, and causal graphs.

## Algorithms

1. **Integrated Gradients**  
   Computes attribution by integrating gradients from a baseline to the input.

2. **Grad-CAM / Grad-CAM++**  
   Heatmaps for CNN/ViT layers.

3. **TCAV**  
   Concept-based interpretability using directional derivatives in hidden layers.

4. **Counterfactual Search**  
   Gradient-based search for minimal input changes to alter predictions.

5. **Causal Graph Estimation**  
   Learns sparse relationships among features/activations and their effect on outputs.

## Examples

See the `examples/` folder for runnable prototypes:

- `integrated_gradients_demo.py`
- `grad_cam_demo.py`
- `tcav_demo.py`
- `counterfactual_demo.py`
- `causal_graph_demo.py`

## Determinism & Scale

- Fix RNG seeds for reproducibility.  
- Use ordered reductions across DP ranks.  
- Overlap path samples with async scheduling.  
- Memory checkpointing + mixed precision supported.

---

This guide is part of the **Tessera documentation suite**.
