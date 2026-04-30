import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    """
    Projection head + prototypes, compatible with DINO/DINOv2-style training.
    """
    def __init__(self, in_dim: int, out_dim: int = 65536, hidden_dim: int = 2048, bottleneck_dim: int = 256, nlayers: int = 3):
        super().__init__()
        layers = []
        dim = in_dim
        for i in range(nlayers - 1):
            layers += [nn.Linear(dim, hidden_dim), nn.GELU(), nn.BatchNorm1d(hidden_dim)]
            dim = hidden_dim
        layers += [nn.Linear(dim, bottleneck_dim)]
        self.mlp = nn.Sequential(*layers)
        self.prototypes = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.prototypes.weight_g.data.fill_(1.0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        logits = self.prototypes(x)
        return logits
