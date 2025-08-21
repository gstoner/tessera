# Tessera QA & Reliability Guide  
## Chapter 7: Stress & Chaos Testing

---

## 7.1 Introduction  

Even when Tessera passes unit and integration tests, large-scale deployments face unpredictable stresses:  
- Long-running workloads  
- Random GPU/node failures  
- Network instability  
- Extreme load and memory pressure  

**Stress & chaos testing** is designed to validate resilience in such environments.  

---

## 7.2 Stress Testing  

Stress testing validates stability under **maximum resource usage** and long runtimes.  

**Scenarios:**  
- Max occupancy kernel launches for hours.  
- Full GPU memory allocation + compute.  
- Mixed precision training under high load.  
- I/O-heavy distributed workloads.  

**Example: Long-Running Stress Test**
```python
import time
from tessera import op, tensor

A = tensor.rand((8192, 8192), dtype="bf16")
B = tensor.rand((8192, 8192), dtype="bf16")

start = time.time()
for i in range(10_000):
    C = op.matmul(A, B)
    if i % 1000 == 0:
        print(f"Iteration {i}, elapsed {time.time() - start:.2f}s")
```

---

## 7.3 Chaos Testing  

Chaos testing injects **controlled failures** to test fault recovery.  
Inspired by Netflix’s *Chaos Monkey* but applied to HPC/AI workloads.  

**Examples of Chaos Events:**  
- Kill a GPU mid-job.  
- Introduce artificial network latency.  
- Simulate ECC memory errors.  
- Randomly drop collectives.  

**Example: Injected Failure**
```python
from tessera import chaos, dist, op

# Run matmul under injected fault
with chaos.inject(fault="kill_device", target=0):
    try:
        Y = op.matmul(A, B)
    except Exception as e:
        print("Recovered from chaos event:", e)
```

---

## 7.4 Distributed Chaos Testing  

For large systems (72–128+ GPUs), chaos testing ensures **collectives and schedulers** can handle failures gracefully.  

**Techniques:**  
- Partition a communication group.  
- Fail one node in a pipeline.  
- Restart a data-parallel worker mid-epoch.  

**Best Practice:** Combine chaos testing with **checkpoint/restart** to validate recovery.  

---

## 7.5 Metrics to Track During Stress & Chaos  

- **Throughput stability** (tokens/sec, images/sec).  
- **Failure recovery time**.  
- **Checkpoint rollback success rate**.  
- **Memory fragmentation levels**.  
- **Deadlock/livelock incidents**.  

---

## 7.6 Best Practices for Stress & Chaos Testing  

- Always run **stress tests before production rollouts**.  
- Use **chaos injection frameworks** in staging clusters.  
- Validate **checkpointing and replay** during chaos events.  
- Automate chaos testing in **CI/CD pipelines for distributed jobs**.  

---
