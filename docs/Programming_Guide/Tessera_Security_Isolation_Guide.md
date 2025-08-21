# Tessera Security & Isolation Guide

## 1. Introduction
This guide defines the **security and isolation guarantees** provided by Tessera when running kernels and distributed workloads across multi-tenant environments.  

The goals are:  
- Provide **safe sharing of GPU resources** across multiple users or jobs.  
- Ensure **memory safety** and deterministic execution semantics.  
- Support **sandboxing** for untrusted kernels and replay-based debugging.  

---

## 2. Multi-Tenant GPU Environments

### 2.1 MIG (Multi-Instance GPU)
- On NVIDIA A100/H100/GB200, Tessera integrates with **MIG partitions**.  
- Tessera’s **mesh abstractions** recognize MIG instances as isolated devices.  
- Each MIG partition is treated as a distinct device ID, with:  
  - Dedicated memory slices.  
  - Fixed compute cores.  
  - QoS guarantees (bandwidth, latency isolation).  

**Example:**
```python
from tessera import dist

mesh = dist.Mesh(devices=["MIG:0:0", "MIG:0:1"])  # 2 MIG instances
```

### 2.2 SR-IOV and vGPU
- On AMD and Intel GPUs, Tessera maps to **SR-IOV virtual functions**.  
- Resource accounting enforced by hardware IOMMU + vendor backend.  
- Distributed collectives detect vGPUs as independent endpoints.  

---

## 3. Memory Safety Guarantees

### 3.1 Address Space Isolation
- Tessera enforces **per-context virtual memory spaces**.  
- Kernel launches cannot access memory outside registered buffers.  
- Illegal memory access → **runtime error** (with diagnosable trace).

### 3.2 Sandboxed Kernels
- JIT-compiled kernels run inside a **sandbox execution context**.  
- Optional **watchdog timers** abort runaway kernels.  
- Memory zeroization on context destruction prevents data leaks.  

### 3.3 Deterministic Replay
- Tessera supports **record/replay mode**:  
  - Record: All random seeds, collectives order, and reduction trees logged.  
  - Replay: Execution re-run with identical scheduling.  
- Use cases: Debugging, regression testing, adversarial input tracing.  

---

## 4. Deterministic Reduction Semantics
- Tessera guarantees that reductions (`all-reduce`, `reduce-scatter`, etc.) are:  
  - **Bitwise deterministic** given identical topology.  
  - **Stable order**: A fixed reduction tree is chosen and persisted.  
- This avoids nondeterministic FP32 accumulation ordering common in CUDA NCCL.  

**Example:**
```python
from tessera import dist, op

with dist.deterministic():
    y = op.all_reduce(x, op="sum")
```

---

## 5. Sandboxing Policies
- **Trusted mode**: Default execution (full optimization enabled).  
- **Sandboxed mode**: Restricts:  
  - Inline PTX  
  - System calls / host callbacks  
  - Out-of-bounds warp shuffles  
- **Strict mode**: For multi-tenant HPC/cloud, ensures all kernels pass bounds checks and disable unsafe intrinsics.  

---

## 6. Debugging & Monitoring
- Tessera integrates with external **GPU profilers** but masks inter-tenant activity.  
- Logs include:  
  - Kernel boundaries  
  - Memory faults  
  - Replay identifiers  

**Replay API:**
```python
import tessera.debug as dbg

dbg.record("run1.json")   # Capture execution trace
...
dbg.replay("run1.json")   # Deterministic re-run
```

---

## 7. Summary
Tessera ensures **strong isolation and security** across multi-tenant GPU deployments:
- MIG/SR-IOV awareness in distributed meshes.  
- Per-context memory isolation & zeroization.  
- Optional sandbox policies for untrusted code.  
- Deterministic collectives & replay for debugging.  

This provides a secure foundation for both **cloud-scale AI inference services** and **multi-tenant HPC workloads**.  
