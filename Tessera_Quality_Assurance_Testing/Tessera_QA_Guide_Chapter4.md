# Tessera QA & Reliability Guide  
## Chapter 4: Cluster-Scale QA (128+ Nodes)

---

## 4.1 Introduction  

At **cluster scale (128+ nodes, thousands of GPUs)**, Tessera QA must ensure correctness, reproducibility, and resilience in highly distributed environments.  
Challenges include:  
- **Hierarchical collectives** across racks.  
- **Checkpoint/restart validation**.  
- **Fault injection for network partitions**.  
- **Weak/strong scaling validation**.  
- **Silent data corruption (SDC) detection**.  

---

## 4.2 Hierarchical Collectives Validation  

Large clusters require **hierarchical collectives** (intra-node NVLink, inter-node InfiniBand).  
QA verifies correctness and performance scaling.  

**Example: Hierarchical All-Reduce**
```python
from tessera import dist, op

# Define a 128-node mesh, each node has 8 GPUs
mesh = dist.Mesh(axes=["node","gpu"], devices=range(1024))

# Perform hierarchical all-reduce
x = dist.tensor([1.0], mesh=mesh, layout=dist.Replicate())
y = op.hierarchical_all_reduce(x, op="sum")

assert y.numpy().tolist() == [1024.0], "Cluster-scale all-reduce failed correctness check"
```

---

## 4.3 Checkpoint/Restart Validation  

At scale, jobs will fail. Tessera supports **fault-tolerant checkpointing**. QA ensures:  
- Checkpoints are consistent across nodes.  
- Restarts reproduce identical results.  

**Example: Checkpoint QA**
```python
from tessera import checkpoint, graph

# Save checkpoint
checkpoint.save("step100", model, optimizer)

# Simulate restart
model2, opt2 = checkpoint.load("step100")

# Ensure weights match
for p1, p2 in zip(model.parameters(), model2.parameters()):
    assert (p1 == p2).all(), "Checkpoint mismatch after restart"
```

---

## 4.4 Network Fault Injection  

Network instability must be tested: packet drops, latency spikes, partitions.  
Tessera provides a **fault injection API** at cluster scale.  

**Example: Simulated Network Partition**
```python
from tessera import fault

with fault.inject(network_partition=[(0,64)]):
    try:
        Y = op.all_gather(X)
    except RuntimeError as e:
        print("Caught simulated network partition:", e)
```

---

## 4.5 Silent Data Corruption (SDC) Detection  

At scale, **silent data corruption** can occur from memory or interconnect faults.  
QA introduces redundant checks (dual-computation, checksums).  

**Example: SDC Detection**
```python
from tessera import verify

Y1 = op.matmul(A, B)
Y2 = op.matmul(A, B, redundant=True)

assert verify.equal(Y1, Y2), "Potential silent data corruption detected"
```

---

## 4.6 Weak and Strong Scaling Tests  

- **Strong scaling**: Fixed problem size, increasing GPUs.  
- **Weak scaling**: Problem size grows with number of GPUs.  

**Best Practice:** Always measure both to detect communication bottlenecks.  

---

## 4.7 Best Practices at Cluster Scale  

- Always validate **hierarchical collectives**.  
- Test **checkpoint/restart reliability** under node failures.  
- Simulate **network partitions and latency spikes**.  
- Use **redundant computations** to detect silent corruption.  
- Perform both **weak and strong scaling benchmarks**.  

---
