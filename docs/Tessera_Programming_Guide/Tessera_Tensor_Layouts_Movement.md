# Tessera Tensor Layouts & Data Movement Guide

## 1. Introduction
Tensors are the core data objects in Tessera. Efficient execution depends not only on compute kernels but also on **layout** and **movement semantics**.  
This guide specifies:
- Supported tensor layouts.
- Formal rules for host↔device and device↔device transfers.
- Asynchronous transfers (`memcpy_async`) and pinned memory.
- Tessera’s **stream/event model**, equivalent to CUDA Streams & Events.

---

## 2. Tensor Layouts

Tessera supports multiple layout strategies, expressed as `LayoutSpec` in the IR.

### 2.1 Row-Major (C-order)
- Contiguous in the last dimension.
- Matches NumPy default (`A[i,j]` stored as `i*stride0 + j`).

### 2.2 Column-Major (Fortran-order)
- Contiguous in the first dimension.
- Useful for LAPACK/BLAS interoperability.

### 2.3 Blocked Layout
- Tensor split into uniform **blocks** (`BM, BN, BK`).
- Each block stored contiguously.
- Maps naturally to **Tile IR** and GPU shared memory.

### 2.4 Tiled Layout
- Generalization of blocked, with **hierarchical tiling** (macro/micro).
- Enables multi-level cache reuse.

### 2.5 Interleaved Layout
- Elements of different tensors interleaved in memory (e.g., structure-of-arrays).
- Used in **fused kernels** or multi-stream pipelines.

---

## 3. Data Movement

### 3.1 Host ↔ Device Transfers
- By default, **host memory is pageable**. For async copies, memory must be **pinned**.
- Transfers may be synchronous or asynchronous.

```python
from tessera import memcpy

# Synchronous host→device copy
memcpy(dst=device_tensor, src=host_array)

# Asynchronous copy on stream
memcpy.async(dst=device_tensor, src=host_array, stream=my_stream)
```

### 3.2 Device ↔ Device Transfers
- Peer-to-peer supported if NVLink/PCIe topology allows.
- Tessera runtime routes copies through best available fabric.

### 3.3 Rules
- Host memory must be **page-locked** for overlap with compute.
- `memcpy_async` requires explicit `stream` binding.
- Overlapping transfers with compute requires **non-default stream**.

---

## 4. Asynchronous Transfers & Streams

### 4.1 Streams
- Tessera streams are **ordered queues of work** (compute + copies).
- Default stream = sequential execution.
- Multiple streams = potential concurrency.

```python
from tessera import stream

s1 = stream.create()
s2 = stream.create()

op.matmul(A,B,stream=s1)
memcpy.async(dst, src, stream=s2)
```

### 4.2 Events
- Events are markers for **synchronization**.
- Can be recorded in one stream and waited on by another.

```python
from tessera import event

ev = event.create()
op.matmul(A,B,stream=s1)
ev.record(s1)

# Stream s2 waits for event
ev.wait(s2)
```

---

## 5. Pinned Memory
- Use `pinned_malloc()` for host arrays involved in async transfers.
- Guarantees page-locking for DMA engines.

```python
from tessera import pinned_malloc

host_buf = pinned_malloc((1024, 1024), dtype="float32")
memcpy.async(device_tensor, host_buf, stream=s1)
```

---

## 6. Overlap & Concurrency Best Practices
- Always use **multiple streams** to overlap copy + compute.
- Use pinned memory for all host↔device async copies.
- Place large batch loads in dedicated **transfer streams**.
- For multi-GPU jobs:  
  - Use **device-to-device async copies** where topology allows.  
  - Fall back to host-staged copies otherwise.

---

## 7. Integration with Schedule IR
At the **Schedule IR** level:
- Layout and movement are expressed in `tensor.layout` and `copy` ops.
- Autotuner may reorder transfers or introduce double-buffering.
- Streams map directly to **Schedule IR pipelines**.

---

## 8. Summary
Tessera tensor movement combines:
- Flexible layouts (row-major, col-major, blocked, tiled, interleaved).
- Explicit, async copy semantics.
- Streams & events for overlap.
- Pinned memory for performance.
- IR-level scheduling that automatically pipelines transfers with compute.
