
# Tessera Probabilistic Programming (TPP)

This document describes the design and prototype implementation of **Probabilistic Programming** in Tessera.

## Goals
- Make uncertainty first-class: random variables, distributions, inference built into the graph.
- Support variational inference (SVI), MAP, and MCMC.
- Shape-safe and deterministic, supporting distributed execution.

## Core API

```python
from tessera.pp import model, sample, observe, Normal

@model
def BayesLinear(x, y):
    w = sample("w", Normal(0, 1))
    sigma = sample("sigma", Normal(0, 1))
    pred = x @ w
    observe("y", Normal(pred, sigma), value=y)
    return pred
```

## Inference

```python
guide = ts.pp.autoguide.mean_field(BayesLinear)
result = ts.pp.svi(BayesLinear, guide, data={"x": X, "y": Y}, steps=5000)
```

## Examples
See `examples/pp/bayes_linear.py` and `examples/pp/bayes_attention.py`.

