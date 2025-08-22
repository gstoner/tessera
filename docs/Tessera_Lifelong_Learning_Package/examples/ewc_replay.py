#!/usr/bin/env python3
import torch, math, random
import torch.nn as nn, torch.nn.functional as F
class TinyNet(nn.Module):
    def __init__(self, d=32, nclass=4):
        super().__init__(); self.enc=nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU()); self.head=nn.Linear(64,nclass)
    def forward(self,x): return self.head(self.enc(x))
def diag_fisher(model, batches=10, B=256, d=32):
    fisher={n:torch.zeros_like(p) for n,p in model.named_parameters() if p.requires_grad}
    for _ in range(batches):
        x=torch.randn(B,d); y=(x[:,:2].sum(-1)>0).long()%4
        logp=F.log_softmax(model(x),-1); sample=torch.distributions.Categorical(logits=logp).sample()
        loss=F.nll_loss(logp, sample); model.zero_grad(); loss.backward()
        for n,p in model.named_parameters():
            if p.grad is not None and p.requires_grad: fisher[n]+=p.grad.detach()**2
    for n in fisher: fisher[n]/=batches
    return fisher
class Reservoir:
    def __init__(self, cap): self.cap=cap; self.store=[]
    def add(self, x, y):
        for i in range(x.size(0)):
            if len(self.store)<self.cap: self.store.append((x[i].cpu(),y[i].cpu()))
            else:
                j=random.randint(0,len(self.store)-1)
                if random.random()< self.cap/(self.cap+1): self.store[j]=(x[i].cpu(),y[i].cpu())
    def sample_like(self, x, ratio=0.25):
        k=min(int(x.size(0)*ratio), len(self.store))
        if k<=0: return x.new_empty((0,x.size(1))), x.new_empty((0,),dtype=torch.long)
        idx=random.sample(range(len(self.store)), k)
        xs=torch.stack([self.store[i][0] for i in idx]).to(x.device)
        ys=torch.stack([self.store[i][1] for i in idx]).to(x.device)
        return xs,ys
def make_batch(B=256,d=32,phase=0.0):
    x=torch.randn(B,d); rot=torch.tensor([[math.cos(phase),-math.sin(phase)],[math.sin(phase),math.cos(phase)]]); x[:,:2]=x[:,:2]@rot; y=((x[:,:2].sum(-1)>0).long()%4)
    return x,y
def train(steps=400, device='cpu'):
    torch.manual_seed(0); random.seed(0); model=TinyNet().to(device); opt=torch.optim.Adam(model.parameters(),lr=2e-3)
    fisher=diag_fisher(model); anchor={n:p.detach().clone() for n,p in model.named_parameters()}; buf=Reservoir(20000); lam=2e-3
    for t in range(steps):
        phase=(t//100)*0.8; x,y=make_batch(256,phase=phase); xr,yr=buf.sample_like(x,0.25); xx=torch.cat([x,xr]).to(device); yy=torch.cat([y,yr]).to(device)
        logits=model(xx); loss=F.cross_entropy(logits,yy)
        reg=0.0
        for (n,p) in model.named_parameters():
            if n in fisher: reg=reg+(lam/2.0)*(fisher[n].to(device)*(p-anchor[n].to(device))**2).sum()
        loss=loss+reg; opt.zero_grad(); loss.backward(); opt.step(); buf.add(x,y)
        if (t+1)%100==0:
            with torch.no_grad():
                xa,ya=make_batch(1024,phase=0.0); xb,yb=make_batch(1024,phase=0.8)
                acc_a=(model(xa.to(device)).argmax(-1)==ya.to(device)).float().mean().item()
                acc_b=(model(xb.to(device)).argmax(-1)==yb.to(device)).float().mean().item()
            print(f'[{t+1:03d}] loss={loss.item():.3f} acc_old={acc_a:.3f} acc_new={acc_b:.3f}')
if __name__=='__main__': train()
