# Tessera Hardware Mapping Guide
## Chapter 3: Programming Model Mapping

Describes how Tessera Graph IR, Schedule IR, and Tile IR map onto GPU thread/block execution.

---

### 3.1 Overview

Tessera introduces a **multi-level IR stack** that maps directly onto GPU hardware.  
This chapter explains the **correspondence between Tessera abstractions and GPU execution units**.

---

### 3.2 Graph IR → Distributed Mesh

- **Graph IR**: expresses computation as an operator graph.  
- Operators (e.g., matmul, FFT) are **device-agnostic**.  
- Tessera introduces the **`Mesh` abstraction**, mapping Graph IR tensors to multi-GPU domains.

Example:
```python
mesh = dist.Mesh(axes=["dp","tp"], devices=range(8))
X = dist.tensor((B, D), layout=dist.ShardSpec(partition=("row",), mesh_axes=("dp",)), mesh=mesh)
```
Mapping:

	•	Graph IR node = computation across one or more GPUs.
	•	Mesh axes = distributed topology (data, tensor, pipeline parallelism).

⸻

3.3 Schedule IR → SMs and Work Distribution

	•	Schedule IR determines how an operator is partitioned.
	•	Maps tiles to Streaming Multiprocessors (SMs).
	•	Handles fusion, tiling, prefetching, and pipelining.

Example (Schedule IR tiling spec):
```mlir
%0 = "tessera.schedule.tile"(%A, %B) {m=128, n=128, k=64}
```

Mapping:

	•	Schedule IR tile (128×128) → one SM block.
	•	Pipeline annotations → prefetch + overlap compute/communication.

⸻

3.4 Tile IR → Warps and Tensor Cores

	•	Tile IR lowers a schedule tile into warp fragments.
	•	Defines explicit mma.sync operations on Tensor Cores.
	•	Warps execute in SIMT, operating on matrix fragments.

 Example (Tile IR MMA):
 ```mlir
%fragC = "tessera.tile.mma_sync"(%fragA, %fragB, %fragC)
          {m=16, n=16, k=16, dtype="bf16"}
```

Mapping:

	•	Warp = executes a fragment multiply-accumulate.
	•	Thread lane = computes one row/col element in MMA fragment.

⸻

3.5 ABI → Target IR

	•	Tessera compiles Tile IR into Target IR (LLVM → PTX/SASS).
	•	ABI ensures consistent lowering across NVIDIA (PTX) and AMD (ROCm LLVM).

Example:

Example (Tile IR MMA):
```llvm
; LLVM IR lowering of tessera.tile.mma_sync
call void @llvm.nvvm.wmma.m16n16k16.load.a.row.stride(...)
```
Mapping:

	•	Tile IR op → vendor-specific intrinsic (mma.sync on NVIDIA).
	•	ABI spec defines tensor layouts and registers per fragment.

⸻

3.6 Multi-GPU Collectives

Graph IR collectives (all-reduce, broadcast, scatter) map to:

	•	NVLink / NVSwitch collectives on NVIDIA.
	•	XGMI / Infinity Fabric collectives on AMD.

Example (Graph IR collective):
```python
Y = dist.all_reduce(X, axis="dp", op="sum")
```
Mapping:

	•	Graph IR collective = runtime NVLink operation.
	•	Schedule IR overlaps collectives with compute.

⸻

3.7 Mapping Table

|Tessera Layer  | GPU Mapping                   |
|---------------|-------------------------------|
|  Graph IR     |Multi-GPU Mesh (NVLink domains)|
| Schedule IR   |SM-level tiling, pipelining    |
|Tile IR        |Warp-level MMA, shared memory  |
|Target IR      | PTX, ROCm LLVM, ISA-specific  |

3.8 Summary

	•	Tessera generalizes CUDA’s grid/block/thread model into multi-level IRs.
	•	Graph IR → distributed GPUs; Schedule IR → SM-level blocks; Tile IR → warps/Tensor Cores.
	•	ABI ensures portability across vendors, while collectives enable efficient scaling.
	•	This mapping provides both clarity (algebraic ops) and performance (hardware-tuned kernels).

⸻

