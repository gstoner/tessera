# Tessera QA & Reliability Guide  
## Chapter 2: Node-Scale QA (Multi-GPU Server)

---

## 2.1 Introduction  

When scaling Tessera from a **single GPU** to a **multi-GPU server**, new QA challenges appear:  
- Ensuring **collective operations** are correct and deterministic.  
- Validating **partitioned execution** under MIG or SR-IOV.  
- Testing **autotuner reproducibility** across devices.  
- Stress-testing with **concurrent workloads**.  

This chapter provides best practices and examples for **node-scale QA**.  

---

## 2.2 Collective Operations Validation  

At node scale, correctness of collectives (`all-reduce`, `all-gather`, `broadcast`) is critical.  
Tessera provides a **collective testing harness** to verify numerical correctness and determinism.  

**Example: All-Reduce Correctness**
```python
from tessera import dist, op, tensor

mesh = dist.Mesh(axes=["dp"], devices=[0,1,2,3])

x = dist.tensor([1.0], mesh=mesh, layout=dist.Replicate())
y = op.all_reduce(x, op="sum")

assert y.numpy().tolist() == [4.0], "All-reduce failed correctness check"
```

---

## 2.3 MIG and SR-IOV Partitioning  

Modern GPUs (A100, H100) support **MIG (Multi-Instance GPU)** and **SR-IOV** for multi-tenant workloads.  
QA at this level ensures Tessera correctly:  
- Recognizes MIG partitions as separate devices.  
- Prevents cross-partition memory leaks.  
- Provides consistent performance isolation.  

**Tip:** Always test kernels inside MIG partitions with stress workloads to validate isolation.  

---

## 2.4 Autotuner Reproducibility  

Tessera’s autotuner selects optimal kernel configs per device. QA should confirm that:  
- Tuned configs are **reproducible across identical hardware**.  
- Persistent caches store results by `(arch, shape, dtype)`.  
- Different devices do not overwrite one another’s caches.  

**Example: Autotuner Test**
```python
from tessera import autotune, op

@autotune.space(configs=[{"block":64}, {"block":128}])
def tuned_matmul(A, B):
    return op.matmul(A, B)

# Run twice and check cached result is reused
Y1 = tuned_matmul(A, B)
Y2 = tuned_matmul(A, B)
assert (Y1 == Y2).all(), "Autotuner inconsistency detected"
```

---

## 2.5 Stress Testing with Concurrent Workloads  

A multi-GPU server often runs **multiple jobs concurrently**. QA tests should simulate contention:  
- Launch multiple Tessera jobs on different GPUs.  
- Validate no interference in results.  
- Monitor memory usage for leaks.  

**Example: Concurrent Jobs**
```python
import multiprocessing as mp
from tessera import op, tensor

def worker(rank):
    A = tensor.rand((1024,1024))
    B = tensor.rand((1024,1024))
    out = op.matmul(A,B)
    print(f"Worker {rank} done.")

if __name__ == "__main__":
    procs = [mp.Process(target=worker, args=(i,)) for i in range(4)]
    [p.start() for p in procs]
    [p.join() for p in procs]
```

---

## 2.6 Best Practices at Node Scale  

- Always validate **collective correctness** across all devices in the node.  
- Run with **deterministic execution** when debugging.  
- Test Tessera kernels inside **MIG partitions** to ensure isolation.  
- Verify **autotuner persistence and reproducibility**.  
- Perform **stress tests with concurrent workloads** to catch edge cases.  

---
