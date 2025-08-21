# Tessera Hardware Mapping Guide
## Chapter 2: GPU Hardware Model

---

### 2.1 Overview

Modern GPUs are massively parallel architectures designed for **throughput computing**.  
Tessera targets these hardware features directly by lowering its IR stack to hardware-level constructs.

Key components:

- **Streaming Multiprocessors (SMs)**  
- **Warps and SIMT execution**  
- **Tensor Cores / MMA units**  
- **Memory hierarchy (HBM, L2, shared, registers)**  
- **GPU interconnects (NVLink, PCIe, NVSwitch)**  

---

### 2.2 Streaming Multiprocessors (SMs)

- Each GPU contains multiple **SMs** (up to 144 on GB200).  
- Each SM schedules **warps of 32 threads**.  
- Tessera’s **Tile IR blocks** are mapped to SM work queues.  

Example mapping:

Schedule IR tile (128×128) → mapped to one SM
Within SM → divided into warp fragments

---

### 2.3 Warps and SIMT

- **Warp** = 32 threads executing in lockstep (SIMT model).  
- Each warp executes **lanes** with different data but the same instruction.  
- Tessera lowers **Tile IR fragments** into warp-level MMA operations.
  
Warp   = SIMD execution group
Lane   = thread executing one element
TileIR = schedules warps to MMA fragments

---

### 2.4 Tensor Cores / MMA Units

- Tensor Cores perform **matrix-matrix multiply-accumulate (MMA)** at high throughput.  
- Operate on FP16, BF16, FP8, INT8, and FP64 (for HPC).  
- Tessera’s **Tile IR** directly encodes **mma.sync** operations.  

MLIR lowering example:
```mlir
%0 = "tessera.tile.mma_sync"(%a, %b, %c)
       {m=16, n=16, k=16, dtype="bf16"}
```
2.5 Memory Hierarchy

GPUs provide multiple memory levels with distinct tradeoffs:

|Memory Level | Latency (cycles) | Bandwidth          | Scope         | Tessera Mapping     |
|-------------|------------------|--------------------|---------------|---------------------|
|Registers    | 1–2              | TB/s               | Per thread    | Tile fragments      |
|Shared Memory| ~20              | 20+ TB/s (per SM)  | Block (SM)    | Tile staging        |
|L2 Cache     | ~200             |~10 TB/s            | All SMs       | Graph IR tensors    |
|HBM3e        |~500–1000         | 8+ TB/s            | Global device | Distributed tensors |

essera’s Schedule IR decides placement:

	•	Prefetch into shared memory for reuse.
	•	Double-buffer between HBM ↔ shared ↔ registers.

⸻

2.6 GPU Interconnects

	•	NVLink 5 (GB200): ~1.8 TB/s bisection bandwidth.
	•	NVSwitch: all-to-all connectivity in DGX/HGX racks.
	•	PCIe Gen5: lower-bandwidth host-device link.

Tessera’s Distributed Mesh API:
```python
mesh = dist.Mesh(axes=["tp","dp"], devices=range(72))
W = dist.tensor((1_000_000, 1_000_000), layout="sharded", mesh=mesh)
```
This maps tensor partitions across NVLink domains.

⸻

2.7 Example Mapping

Matrix multiply on a GPU:

	1.	Graph IR: C = A × B
	2.	Schedule IR: tile into (128×128×64) blocks
	3.	Tile IR: assign warps to Tensor Cores
	4.	Target IR: emit mma.sync PTX ops
	5.	Execution:
	•	Registers hold fragments
	•	Shared memory stages tiles
	•	HBM provides global operands

⸻

2.8 Summary

	•	GPUs provide hierarchical parallelism: SMs → warps → lanes → Tensor Cores.
	•	Tessera lowers IRs to these hardware constructs seamlessly.
	•	Memory hierarchy and interconnects are first-class in Schedule IR and Distributed Mesh.
	•	This mapping allows Tessera to scale from a single GPU to NVLink-connected clusters.

⸻

