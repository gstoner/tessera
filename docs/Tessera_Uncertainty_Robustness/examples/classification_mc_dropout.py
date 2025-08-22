#!/usr/bin/env python3
import torch, torch.nn as nn, torch.nn.functional as F

class Clf(nn.Module):
    def __init__(self, d, k, pdrop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(), nn.Dropout(pdrop),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(pdrop),
            nn.Linear(128, k)
        )
    def forward(self, x): return self.net(x)

@torch.no_grad()
def classify_with_uncertainty(model, x, samples=30):
    was_training = model.training
    model.train()  # enable dropout
    probs = []
    for _ in range(samples):
        logits = model(x)
        probs.append(F.softmax(logits, dim=-1))
    P = torch.stack(probs, 0)   # [S,B,K]
    p_mean = P.mean(0)
    total_unc = -(p_mean * (p_mean.clamp_min(1e-8)).log()).sum(-1)
    ale = (-(P * P.clamp_min(1e-8).log()).sum(-1)).mean(0)
    epi = total_unc - ale
    if not was_training: model.eval()
    return p_mean, epi, ale

def demo():
    torch.manual_seed(0)
    x = torch.randn(512, 16); y = (x[:,0]>0).long()
    m = Clf(16, 2); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for _ in range(400):
        m.train(); loss = F.cross_entropy(m(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    p, epi, ale = classify_with_uncertainty(m, x[:8], samples=50)
    print('p:', p.tolist())
    print('epistemic:', epi.tolist())
    print('aleatoric:', ale.tolist())

if __name__ == '__main__':
    demo()
