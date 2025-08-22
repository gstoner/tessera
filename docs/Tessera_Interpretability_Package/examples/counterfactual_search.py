#!/usr/bin/env python3
import torch, torch.nn as nn, torch.nn.functional as F

def counterfactual_search(model, x0, target_idx, l1=0.01, l2=0.01, steps=300, lr=0.05, box=None):
    x_cf = x0.clone().detach().requires_grad_(True)
    opt = torch.optim.SGD([x_cf], lr=lr)
    targ = torch.full((x_cf.size(0),), target_idx, dtype=torch.long, device=x_cf.device)
    lo, hi = (None, None)
    if box is not None:
        lo, hi = box
    for _ in range(steps):
        logits = model(x_cf)
        ce = F.cross_entropy(logits, targ)
        prox = l1 * x_cf.abs().sum() + l2 * (x_cf**2).sum()
        loss = ce + prox
        opt.zero_grad(); loss.backward(); opt.step()
        if lo is not None and hi is not None:
            with torch.no_grad(): x_cf.clamp_(lo, hi)
    dx = x_cf.detach() - x0.detach()
    return x_cf.detach(), dx

# Tiny demo with a linear classifier
class Tiny(nn.Module):
    def __init__(self,d=8,k=2):
        super().__init__(); self.w=nn.Linear(d,k,bias=False)
    def forward(self,x): return self.w(x)

if __name__ == "__main__":
    torch.manual_seed(0)
    m = Tiny()
    x = torch.randn(4,8)
    x_cf, dx = counterfactual_search(m, x, target_idx=1, box=(-2.0, 2.0))
    print("Δx norms:", dx.norm(dim=1).tolist())
