#!/usr/bin/env python3
import torch, torch.nn as nn, torch.nn.functional as F

class Pred:
    def __init__(self, mean, std, epistemic=None, aleatoric=None):
        self.mean = mean; self.std = std
        self.epistemic = epistemic; self.aleatoric = aleatoric

class HetReg(nn.Module):
    def __init__(self, d, hidden=128, pdrop=0.2):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(pdrop),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(pdrop)
        )
        self.mu = nn.Linear(hidden, 1)
        self.logvar = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self.f(x)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-10, 5)
        return mu, logvar

def gaussian_nll(mu, logvar, y):
    inv = torch.exp(-logvar)
    return 0.5*(inv*(y-mu)**2 + logvar).mean()

@torch.no_grad()
def predict_with_uncertainty(model, x, samples=30, train_mode=True):
    was_training = model.training
    if train_mode: model.train()
    else: model.eval()
    mus, sig2s = [], []
    for _ in range(samples):
        mu, logvar = model(x)
        mus.append(mu[...,0])
        sig2s.append(torch.exp(logvar[...,0]))
    mus = torch.stack(mus, 0)
    sig2s = torch.stack(sig2s, 0)
    mu_hat = mus.mean(0)
    aleatoric = sig2s.mean(0)
    epistemic = mus.var(0, unbiased=False)
    std = (aleatoric + epistemic).sqrt()
    if was_training: model.train()
    return Pred(mu_hat, std, epistemic, aleatoric)

def demo():
    torch.manual_seed(0)
    x = torch.randn(1024, 8); y = (2.0*x[:,0] + 0.3*torch.randn_like(x[:,0])).unsqueeze(-1)
    m = HetReg(8); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for _ in range(500):
        mu, logv = m(x); loss = gaussian_nll(mu, logv, y)
        opt.zero_grad(); loss.backward(); opt.step()
    pred = predict_with_uncertainty(m, x[:5], samples=50)
    print('mean:', pred.mean.tolist())
    print('std:', pred.std.tolist())
    print('epistemic:', pred.epistemic.tolist())
    print('aleatoric:', pred.aleatoric.tolist())

if __name__ == '__main__':
    demo()
