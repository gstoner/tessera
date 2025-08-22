import torch, torch.nn as nn, torch.nn.functional as F

class Clf(nn.Module):
    def __init__(self, d, k, pdrop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(), nn.Dropout(pdrop),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(pdrop),
            nn.Linear(128, k)
        )
    def forward(self, x): return self.net(x)
