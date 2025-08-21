# Tessera Example — Spectral Mixture‑of‑Experts (MoE)

**Goal:** Combine spectral transforms (FFT), recursive operators, and MoE routing to build a scalable, energy‑efficient architecture on Tessera.

---

## 1) Architecture Sketch

- Input → **FFT** → Spectral tokens
- **Experts**: small MLPs operating per‑frequency band
- **Routing**: learned scores per token → top‑k experts
- **Recursive composition**: mixture applied multiple times (depth L)
- Inverse FFT to return to spatial domain

---

## 2) Tessera Implementation

```python
from tessera import op, dist

def spectral_moe(x, num_experts=16, topk=2, depth=3):
    Xf = op.fft(x)  # [B, N, D] → spectral
    experts = [op.mlp(in_dim=D, hidden=[4096], out_dim=D, activation="gelu")
               for _ in range(num_experts)]

    router = op.linear(D, num_experts)   # routing scores per token

    def mixture(z):
        scores = router(z)               # [B,N,E]
        idx, gate = op.topk(scores, k=topk, softmax=True)  # sparse gating
        return op.moe_apply(z, experts, idx, gate)

    z = Xf
    for _ in range(depth):
        z = mixture(z)

    return op.ifft(z)
```

---

## 3) Distribution & Sharding

```python
from tessera import dist

mesh = dist.Mesh(axes=["dp","ep"], devices=range(64))

# Expert parallel (ep): shard experts across devices
expert_shard = dist.ShardSpec(partition=("experts",), mesh_axes=("ep",))

# Data parallel (dp): shard batch
x = dist.tensor((B, N, D), layout=dist.ShardSpec(("row",), ("dp",)), mesh=mesh)

y = spectral_moe(x)
```

- Router replicated on dp; experts sharded on ep
- Sparse all‑to‑all only for selected tokens → reduced comms

---

## 4) Scheduling & Optimization

```python
from tessera import schedule

@schedule(pipeline=True, tile={"BM":128, "BN":128, "BK":32}, warps=8)
def train_step(batch):
    return spectral_moe(batch["x"])
```

- FFT lowered to tile‑wise kernels with collectives for large N
- MoE uses **sparse collectives** and **hierarchical routing** per sub‑mesh

---

## 5) Notes

- Energy‑efficient: spectral domain reduces compute per expert
- Deterministic: fixed reduction order for expert aggregation
- Elastic: experts re‑hashed with consistent hashing on resize
