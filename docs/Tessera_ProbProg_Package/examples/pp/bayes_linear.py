#!/usr/bin/env python3
"""Bayesian linear regression demo (prototype)."""
import math, random
from tessera.pp import model, sample, observe, Normal, HalfCauchy, autoguide, svi

def dot(x, w):
    if isinstance(w, (list, tuple)):
        return sum(xi*wi for xi,wi in zip(x, w))
    return x*w

@model
def BayesLinear(x, y=None):
    # Vector weight prior (match x dimension)
    D = len(x[0]) if isinstance(x[0], (list, tuple)) else 1
    w = sample("w", Normal([0.0]*D, [1.0]*D))
    sigma = sample("sigma", HalfCauchy(1.0))
    for i,(xi, yi) in enumerate(zip(x,y or [None]*len(x))):
        pred = dot(xi, w)
        if y is not None:
            observe(f"y{i}", Normal(pred, sigma), value=yi)
    return {"w": w, "sigma": sigma}

def synth_data(n=64, d=3, noise=0.1, seed=0):
    r = random.Random(seed)
    w_true = [r.uniform(-1,1) for _ in range(d)]
    xs, ys = [], []
    for _ in range(n):
        x = [r.uniform(-1,1) for _ in range(d)]
        y = sum(a*b for a,b in zip(x, w_true)) + r.gauss(0, noise)
        xs.append(x); ys.append(y)
    return xs, ys, w_true

if __name__ == "__main__":
    X, Y, w_true = synth_data(n=64, d=3, noise=0.2, seed=42)
    guide = autoguide.mean_field(BayesLinear)
    res = svi(BayesLinear, guide, data={"x": X, "y": Y}, steps=200, lr=0.05, seed=1)
    print("Posterior:", res["posterior"])
    print("True w:", w_true)
