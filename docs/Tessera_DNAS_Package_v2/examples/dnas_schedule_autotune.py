#!/usr/bin/env python3
import json, os, math, random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

def gumbel_softmax_sample(logits, tau=1.0, eps=1e-10):
    U = torch.rand_like(logits)
    g = -torch.log(-torch.log(U + eps) + eps)
    y = (logits + g) / max(tau, 1e-6)
    return F.softmax(y, dim=-1)

def flops_linear(in_features, out_features, batch):
    return 2.0 * batch * in_features * out_features

def bytes_linear(in_features, out_features, batch, dtype_bytes=2):
    act = batch * in_features * dtype_bytes
    wts = in_features * out_features * dtype_bytes
    out = batch * out_features * dtype_bytes
    return act + wts + out

class LinearBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d_in = d; self.d_out = d
        self.fc = nn.Linear(d, d, bias=False)
    def forward(self, x): return self.fc(x)

class GatedMLP(nn.Module):
    def __init__(self, d, expansion=4):
        super().__init__()
        self.d = d; self.expansion = expansion
        self.fc1 = nn.Linear(d, d*expansion, bias=False)
        self.fc2 = nn.Linear(d*expansion, d, bias=False)
    def forward(self, x): return self.fc2(F.gelu(self.fc1(x)))

class MixedOp(nn.Module):
    def __init__(self, d, temperature=4.0):
        super().__init__()
        self.candidates = nn.ModuleList([LinearBlock(d), GatedMLP(d, 4)])
        self.alpha = nn.Parameter(torch.zeros(len(self.candidates)))
        self.temperature = temperature
    def forward(self, x):
        gate = gumbel_softmax_sample(self.alpha, tau=self.temperature)
        outs = torch.stack([op(x) for op in self.candidates], dim=0)  # [K,B,D]
        y = (gate.view(-1,1,1) * outs).sum(dim=0)
        return y, gate

class ScheduleSpace(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha_tm = nn.Parameter(torch.zeros(2))
        self.alpha_tn = nn.Parameter(torch.zeros(2))
        self.alpha_tk = nn.Parameter(torch.zeros(2))
        self.alpha_st = nn.Parameter(torch.zeros(3))
        self.temperature = 4.0
    def probs(self):
        tau = self.temperature
        return {
            "tm": gumbel_softmax_sample(self.alpha_tm, tau=tau),
            "tn": gumbel_softmax_sample(self.alpha_tn, tau=tau),
            "tk": gumbel_softmax_sample(self.alpha_tk, tau=tau),
            "st": gumbel_softmax_sample(self.alpha_st, tau=tau),
        }
    def argmax_choices(self):
        tm = [64,128][torch.argmax(self.alpha_tm).item()]
        tn = [128,256][torch.argmax(self.alpha_tn).item()]
        tk = [32,64][torch.argmax(self.alpha_tk).item()]
        st = [2,3,4][torch.argmax(self.alpha_st).item()]
        return {"tile_m": tm, "tile_n": tn, "tile_k": tk, "stages": st}

class CostModel:
    def __init__(self, cache_path):
        self.cache_path = Path(cache_path)
        self.cache = {}
        if self.cache_path.exists():
            try: self.cache = json.loads(self.cache_path.read_text())
            except Exception: self.cache = {}
    def predict(self, flops, bytes_, probs_sched):
        tm = probs_sched["tm"] @ torch.tensor([64.0,128.0]).to(flops)
        tn = probs_sched["tn"] @ torch.tensor([128.0,256.0]).to(flops)
        tk = probs_sched["tk"] @ torch.tensor([32.0,64.0]).to(flops)
        st = probs_sched["st"] @ torch.tensor([2.0,3.0,4.0]).to(flops)
        tile_pen = 1.0/(tm*tn).clamp(min=1.0) + 0.2/tk.clamp(min=1.0)
        stage_pen = 0.05*st
        lat = 1e-12*flops + 1e-9*bytes_ + 5e-3*tile_pen + stage_pen*1e-3
        energy = 4e-12*flops + 2e-9*bytes_ + 2e-3*tile_pen
        mem = bytes_
        return lat, energy, mem
    def measure_and_update(self, key, lat_pred):
        import random
        noise = random.uniform(-0.05, 0.05) * float(lat_pred.item())
        measured = max(0.0, float(lat_pred.item()) + noise)
        self.cache[key] = measured
        self.cache_path.write_text(json.dumps(self.cache, indent=2))
        return measured

class TinyNASWithSched(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.embed = nn.Linear(d, d)
        self.block = MixedOp(d)
        self.head  = nn.Linear(d, 4)
        self.sched = ScheduleSpace()
    def forward(self, x):
        h = F.gelu(self.embed(x))
        h, gate = self.block(h)
        y = self.head(h)
        return y, gate
    def weight_parameters(self):
        return [p for n,p in self.named_parameters() if "alpha" not in n]
    def arch_parameters(self):
        return [p for n,p in self.named_parameters() if "block.alpha" in n]
    def sched_parameters(self):
        return [p for n,p in self.named_parameters() if "sched.alpha" in n]

def run(steps=200, batch=512, device="cpu", cache="tessera_autotune_cache.json"):
    torch.manual_seed(123)
    model = TinyNASWithSched().to(device)
    cm = CostModel(cache)
    opt_w = torch.optim.Adam(model.weight_parameters(), lr=3e-3)
    opt_a = torch.optim.Adam(model.arch_parameters(),   lr=5e-2)
    opt_s = torch.optim.Adam(model.sched_parameters(),  lr=5e-2)

    def batch_data(B):
        x = torch.randn(B, 128, device=device)
        y = (x[:, :4].sum(dim=-1) > 0).long() % 4
        return x, y
    xtr,ytr = batch_data(batch)
    xva,yva = batch_data(batch)

    tau0 = 4.0
    for t in range(steps):
        model.block.temperature = max(0.3, tau0 * (1 - t/max(1,steps-1)) + 0.3)
        model.sched.temperature = max(0.3, tau0 * (1 - t/max(1,steps-1)) + 0.3)

        # weights
        model.train()
        x,y = batch_data(batch)
        logits, _ = model(x)
        loss_w = torch.nn.functional.cross_entropy(logits, y)
        opt_w.zero_grad(); loss_w.backward(); opt_w.step()

        # architecture + schedule
        if t % 2 == 0:
            model.eval()
            with torch.no_grad():
                logits_v, _ = model(xva)
                task_v = torch.nn.functional.cross_entropy(logits_v, yva)
            # proxy cost
            probs_sched = model.sched.probs()
            probs_arch = torch.softmax(model.block.alpha, dim=-1)
            f_lin = flops_linear(128,128,batch); b_lin = bytes_linear(128,128,batch)
            f_gml = flops_linear(128,128*4,batch) + flops_linear(128*4,128,batch)
            b_gml = bytes_linear(128,128*4,batch) + bytes_linear(128*4,128,batch)
            flops = probs_arch @ torch.tensor([f_lin, f_gml], device=device, dtype=torch.float32)
            bytes_ = probs_arch @ torch.tensor([b_lin, b_gml], device=device, dtype=torch.float32)
            lat, en, mem = cm.predict(flops, bytes_, probs_sched)
            if t % 25 == 0:
                _ = cm.measure_and_update(f"B{batch}-D128", lat)
            loss_a = task_v + 1e-3*lat + 1e-4*en + 1e-5*mem

            opt_a.zero_grad(); opt_s.zero_grad()
            logits_v2, _ = model(xva)
            task_v2 = torch.nn.functional.cross_entropy(logits_v2, yva)
            total = 0.7*task_v2 + 0.3*(1e-3*lat + 1e-4*en + 1e-5*mem)
            total.backward()
            opt_a.step(); opt_s.step()

        if (t+1) % 50 == 0:
            with torch.no_grad():
                pa = torch.softmax(model.block.alpha, dim=-1).cpu().numpy().tolist()
                ps = {k: v.detach().cpu().numpy().tolist() for k,v in model.sched.probs().items()}
            print(f"[{t+1:03d}] loss_w={loss_w.item():.3f} arch_probs={pa} tm/tn/tk/st={ps['tm']}/{ps['tn']}/{ps['tk']}/{ps['st']}")

    arch_idx = torch.argmax(model.block.alpha).item()
    sched = model.sched.argmax_choices()
    print("Chosen arch idx:", arch_idx, "(0=Linear,1=GatedMLP)")
    print("Chosen schedule:", sched)

if __name__ == "__main__":
    run()
