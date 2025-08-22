import torch, torch.nn as nn, torch.nn.functional as F

def counterfactual_search(model, x, target_idx, steps=200, lr=0.05):
    x = x.clone().detach().requires_grad_(True)
    opt = torch.optim.SGD([x], lr=lr)
    for _ in range(steps):
        logits = model(x)
        ce = F.cross_entropy(logits, torch.full((x.size(0),), target_idx, dtype=torch.long))
        loss = ce + 0.01*(x.abs().sum()) + 0.01*((x**2).sum())
        opt.zero_grad(); loss.backward(); opt.step()
    return x.detach()

class SimpleMLP(nn.Module):
    def __init__(self,d=16,k=2): super().__init__(); self.fc1=nn.Linear(d,32); self.fc2=nn.Linear(32,k)
    def forward(self,x): return self.fc2(F.relu(self.fc1(x)))

if __name__ == "__main__":
    torch.manual_seed(0)
    m=SimpleMLP()
    x=torch.randn(4,16)
    cf=counterfactual_search(m,x,target_idx=1)
    print("Counterfactual:", cf.shape)
