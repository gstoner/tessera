#!/usr/bin/env python3
import torch, torch.nn as nn, torch.nn.functional as F

def integrated_gradients(model, x, baseline=None, steps=64, target_idx=None):
    device = x.device
    if baseline is None: baseline = torch.zeros_like(x)
    alphas = torch.linspace(0, 1, steps, device=device).view(-1, *([1]*(x.dim())))
    path = baseline + alphas*(x - baseline)       # [steps, B, ...]
    path.requires_grad_(True)
    logits = model(path)                           # [steps, B, C] or [steps, B, 1]
    out = logits[..., target_idx] if target_idx is not None else logits.squeeze(-1)
    grads = torch.autograd.grad(out.sum(), path)[0]
    avg_grad = grads.mean(dim=0)
    ig = (x - baseline) * avg_grad
    return ig

class MLP(nn.Module):
    def __init__(self,d=16,k=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,64), nn.ReLU(), nn.Linear(64,k))
    def forward(self,x): return self.net(x)

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(64, 16); y = (x[:,0]>0).long()
    m = MLP(16,2); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for _ in range(300):
        loss = F.cross_entropy(m(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    ig = integrated_gradients(m, x[:4], steps=64, target_idx=1)
    print("IG shape:", ig.shape)
    print("Top-5 dims by |IG| mean:", ig.abs().mean(0).topk(5).indices.tolist())
