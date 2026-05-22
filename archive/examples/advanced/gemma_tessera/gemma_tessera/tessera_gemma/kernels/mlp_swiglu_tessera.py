import torch, torch.nn as nn

class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.wi = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.wo = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        x12 = self.wi(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.wo(torch.nn.functional.silu(x1) * x2)
