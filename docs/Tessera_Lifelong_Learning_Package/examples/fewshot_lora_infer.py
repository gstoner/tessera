#!/usr/bin/env python3
import torch, math
import torch.nn as nn, torch.nn.functional as F
class LoRA(nn.Module):
    def __init__(self, base: nn.Linear, rank=8, alpha=16.0):
        super().__init__(); self.base=base; self.A=nn.Parameter(torch.zeros(base.in_features,rank)); self.B=nn.Parameter(torch.zeros(rank,base.out_features)); self.scaling=alpha/rank
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5)); nn.init.zeros_(self.B)
        for p in self.base.parameters(): p.requires_grad_(False)
    def forward(self, x): return x@self.base.weight.T + (x@self.A)@self.B*self.scaling
class Model(nn.Module):
    def __init__(self,d=64,nclass=4,rank=8):
        super().__init__(); self.fc1=nn.Linear(d,d,bias=False); self.lora=LoRA(self.fc1,rank); self.fc2=nn.Linear(d,nclass,bias=False)
    def forward(self,x): return self.fc2(F.gelu(self.lora(x)))
def fit(delta_model, xs, ys, steps=40, lr=5e-3):
    params=[p for n,p in delta_model.named_parameters() if 'A' in n or 'B' in n]
    opt=torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        loss=F.cross_entropy(delta_model(xs), ys); opt.zero_grad(); loss.backward(); opt.step()
def demo():
    torch.manual_seed(0); m=Model()
    xs=[]; ys=[]
    for c in range(4):
        x=torch.randn(3,64)+ (c*0.5); y=torch.full((3,),c,dtype=torch.long); xs.append(x); ys.append(y)
    xs=torch.cat(xs); ys=torch.cat(ys)
    fit(m, xs, ys, steps=60, lr=5e-3)
    q=torch.randn(16,64)+0.75; print('pred:', m(q).argmax(-1).tolist())
if __name__=='__main__': demo()
