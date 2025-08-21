# Tessera QA & Reliability Guide  
## Chapter 3: Rack-Scale QA (NVL72 / 72 GPUs)

---

## 3.1 Introduction  

When scaling Tessera to a **rack-scale system** such as an **NVIDIA NVL72 (72 GPUs)**, QA focuses on:  
- **Topology-aware correctness** of distributed collectives.  
- **Pipeline, tensor, and data parallelism validation** across the full rack.  
- **Deterministic reductions** at scale.  
- **Bandwidth stress testing** for NVLink fabric.  
- **Failure injection testing** (device loss, network instability).  

This chapter provides practical guidelines and test patterns for **rack-scale validation**.  

---

## 3.2 Mesh Layout Testing  

At rack scale, Tessera relies on **multi-dimensional meshes** (e.g., tensor × pipeline × data parallel).  

**Example: Mesh Layout QA**
```python
from tessera import dist, tensor

# Define 6×6×2 mesh over 72 GPUs
mesh = dist.Mesh(axes=["tp","pp","dp"], devices=range(72))

# Validate tensor sharding
W = dist.tensor((1_000,1_000),
    layout=dist.ShardSpec(partition=("row","col"), mesh_axes=("tp","pp")),
    mesh=mesh)

assert W.sharding_valid(), "Sharding mismatch detected at rack scale"
```

---

## 3.3 Deterministic Reductions  

Large-scale reductions (e.g., all-reduce) must remain **deterministic**, even with 72 participants.  

**Example: Deterministic All-Reduce**
```python
from tessera import dist, op

mesh = dist.Mesh(axes=["dp"], devices=range(72))

x = dist.tensor([1.0], mesh=mesh, layout=dist.Replicate())
y = op.all_reduce(x, op="sum", deterministic=True)

assert y.numpy().tolist() == [72.0], "Rack-scale all-reduce failed determinism check"
```

---

## 3.4 Bandwidth Stress Testing  

At rack scale, **NVLink-Switch fabric bandwidth** becomes the bottleneck.  
QA should include stress tests to ensure collectives achieve **expected throughput (~1.8TB/s bisection)**.  

**Example: Bandwidth Test**
```python
from tessera import dist, op, tensor, profile

mesh = dist.Mesh(axes=["dp"], devices=range(72))

X = dist.tensor((4096,4096), mesh=mesh)
with profile.session() as sess:
    Y = op.all_reduce(X, op="sum")
    sess.report()  # Reports achieved bandwidth and latency
```

---

## 3.5 Failure Injection Testing  

Rack-scale systems experience failures: GPU drops, NVLink instability, ECC errors.  
Tessera provides a **failure injection framework** for QA.  

**Example: Injected GPU Failure**
```python
from tessera import fault

mesh = dist.Mesh(axes=["dp"], devices=range(72))

with fault.inject(drop_device=5):
    try:
        Y = op.all_reduce(X, op="sum")
    except RuntimeError as e:
        print("Caught simulated GPU failure:", e)
```

---

## 3.6 Best Practices at Rack Scale  

- Validate **multi-axis mesh sharding** matches design.  
- Run **determinism tests** across all 72 devices.  
- Perform **NVLink bandwidth stress tests**.  
- Use **failure injection** to ensure recovery mechanisms.  
- Regularly repeat tests under **production workloads**.  

---
