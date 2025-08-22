import torch, torch.nn as nn

class Pred:
    def __init__(self, mean, std, epistemic=None, aleatoric=None):
        self.mean, self.std = mean, std
        self.epistemic, self.aleatoric = epistemic, aleatoric

class HetReg(nn.Module):
    def __init__(self, d, hidden=128, pdrop=0.1):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(),
            nn.Dropout(pdrop),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(pdrop)
        )
        self.mu = nn.Linear(hidden, 1)
        self.logvar = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.f(x)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-10, 5)
        return mu, logvar
