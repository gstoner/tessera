import torch, torch.nn as nn, torch.nn.functional as F

class FeatNet(nn.Module):
    def __init__(self,d=16,k=2):
        super().__init__()
        self.f1 = nn.Linear(d,64); self.f2 = nn.Linear(64,64); self.head = nn.Linear(64,k)
    def features(self,x): return F.relu(self.f2(F.relu(self.f1(x))))
    def forward(self,x): return self.head(self.features(x))

def tcav_score(model, concept_x, random_x, x_eval, target_idx, layer_fn):
    with torch.no_grad():
        Hc = layer_fn(concept_x); Hr = layer_fn(random_x)
    cav = (Hc.mean(0) - Hr.mean(0)); cav = cav / (cav.norm()+1e-8)
    x_eval.requires_grad_(True)
    H = layer_fn(x_eval)
    eps=1e-2
    dpos = model.head(H + eps*cav).sum(dim=1)
    dneg = model.head(H - eps*cav).sum(dim=1)
    dir_deriv = (dpos - dneg) / (2*eps)
    return (dir_deriv>0).float().mean().item()

if __name__ == "__main__":
    torch.manual_seed(0)
    m = FeatNet()
    concept = torch.randn(64,16) + 1.0
    randoms = torch.randn(64,16)
    evalx   = torch.randn(128,16)
    print("TCAV score:", tcav_score(m, concept, randoms, evalx, target_idx=1, layer_fn=m.features))
