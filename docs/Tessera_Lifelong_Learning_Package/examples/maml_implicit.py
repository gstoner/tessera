#!/usr/bin/env python3
import torch, random
import torch.nn as nn, torch.nn.functional as F
class Tiny(nn.Module):
    def __init__(self,d=16,n=2): super().__init__(); self.w=nn.Linear(d,n,bias=False)
    def forward(self,x): return self.w(x)
def task_batch(k=5,d=16,shift=0.0):
    x=torch.randn(k,d)+shift; y=(x[:,0]>0).long(); return x,y
def run(steps=80):
    torch.manual_seed(0); random.seed(0); m=Tiny(); opt=torch.optim.SGD(m.parameters(), lr=0.1)
    for t in range(steps):
        xs,ys=task_batch(8,shift=0.0); xq,yq=task_batch(8,shift=0.5)
        fast=Tiny(); fast.load_state_dict(m.state_dict())
        for _ in range(3):
            loss=F.cross_entropy(fast(xs), ys); g=torch.autograd.grad(loss, fast.parameters(), create_graph=True)
            with torch.no_grad():
                for p,gi in zip(fast.parameters(), g): p.add_(-0.05*gi)
        outer=F.cross_entropy(fast(xq), yq)
        v=torch.autograd.grad(outer, fast.parameters(), retain_graph=True)
        lam=0.1; meta=[vi/(1+lam) for vi in v]
        g_model=torch.autograd.grad(list(fast.parameters()), list(m.parameters()), grad_outputs=meta)
        for p,gi in zip(m.parameters(), g_model): p.grad=gi
        opt.step()
        if (t+1)%20==0: print(f'[{t+1:03d}] outer={outer.item():.3f}')
    print('Done.')
if __name__=='__main__': run()
