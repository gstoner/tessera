#!/usr/bin/env python3
import torch

@torch.no_grad()
def conformal_calibrate(predict_fn, x_cal, y_cal, samples=30):
    pred = predict_fn(x_cal, samples=samples)
    r = (y_cal.squeeze(-1) - pred.mean).abs() / pred.std.clamp_min(1e-6)
    return r.sort().values

@torch.no_grad()
def conformal_interval(predict_fn, x, cal_scores, alpha=0.1, samples=30):
    n = len(cal_scores)
    q_idx = int((1 - alpha) * (n + 1)) - 1
    q_idx = max(0, min(q_idx, n-1))
    q = cal_scores[q_idx]
    pred = predict_fn(x, samples=samples)
    lo = pred.mean - q*pred.std
    hi = pred.mean + q*pred.std
    return lo, hi
