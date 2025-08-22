@torch.no_grad()
def conformal_calibrate(model, x_cal, y_cal, samples=30):
    pred = predict_with_uncertainty(model, x_cal, samples=samples)
    r = (y_cal.squeeze(-1) - pred.mean).abs() / pred.std.clamp_min(1e-6)
    return r.sort().values
