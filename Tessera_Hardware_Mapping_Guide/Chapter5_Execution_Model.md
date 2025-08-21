# Tessera Hardware Mapping Guide
## Chapter 5: Execution Model

Describes kernel launches, pipelines, operator fusion, streams, and determinism.

---

### 5.1 Overview

The **execution model** defines how Tessera programs are launched and scheduled on GPU hardware.  
Unlike CUDA’s grid/block/thread model, Tessera relies on **multi-level IRs** that determine execution mapping:

- **Graph IR**: high-level operators and collectives.  
- **Schedule IR**: tiling, fusion, and pipelining at the SM/block level.  
- **Tile IR**: warp- and Tensor Core-level instructions.  
- **Target IR**: hardware-specific instructions (PTX, ROCm LLVM).  

---

### 5.2 Kernel Launch Semantics

- Tessera lowers a **Graph IR operator** into one or more **GPU kernels**.  
- Each kernel contains:
  - Tiling (Schedule IR) → maps to SM blocks.  
  - Warp decomposition (Tile IR) → maps to warps and Tensor Cores.  
- Launches are **runtime-managed**: persistent kernels, streams, or fused operators.  

Example (Python API):
```python
@graph.kernel
def gemm(A, B, C):
    C[:] = op.matmul(A, B)
```
This lowers into:

	•	Graph IR: matmul node.
	•	Schedule IR: tiling (128×128).
	•	Tile IR: warp-level MMA fragments.

⸻

5.3 Scheduling Policies

Tessera’s Schedule IR provides policies for operator execution:

	•	Greedy tiling: maximize SM occupancy.
	•	Pipeline scheduling: overlap load/compute/store.
	•	Fusion: merge adjacent ops into single kernels.
	•	Collective overlap: interleave comm + compute.

MLIR annotation:
```mlir
%0 = "tessera.schedule.pipeline"(%A, %B) {double_buffer=true}
```

5.4 Pipeline Execution

Tessera supports multi-stage pipelines:

	•	Stage 1: Prefetch inputs from HBM → shared.
	•	Stage 2: Load into registers.
	•	Stage 3: Compute on Tensor Cores.
	•	Stage 4: Writeback to HBM.

Pattern:
```
load (N+1), compute (N), store (N-1)
```

5.5 Operator Fusion

Fusion combines multiple Graph IR ops into one kernel:

	•	Example: Y = relu(matmul(A, B) + bias)
	•	Instead of 3 kernels, Tessera fuses into one: load A/B, compute matmul, add bias, apply relu.

Benefits:

	•	Reduced memory traffic.
	•	Fewer kernel launches.
	•	Better Tensor Core utilization.

⸻

5.6 Multi-GPU Execution

Tessera schedules operators across GPUs using distributed meshes:

	•	Each GPU executes local tiles.
	•	Collectives synchronize partial results.
	•	Pipelines overlap compute + communication.

Example:
```python
Y = dist.all_reduce(op.matmul(A, B), axis="dp", op="sum")
```
Execution:

	1.	Local matmul per GPU.
	2.	All-reduce across data-parallel mesh.
	3.	Writeback results.

⸻

5.7 Streams and Overlap

Tessera runtime supports:

	•	Multiple streams for overlapping independent kernels.
	•	Graph execution: reorder-ready DAG scheduling.
	•	Event synchronization: precise dependency management.

```python
with graph.stream("compute"):
    out = op.matmul(A, B)
with graph.stream("comm"):
    out = dist.all_reduce(out, axis="dp")
```

5.8 Deterministic Execution

	•	Tessera enforces fixed reduction order for collectives.
	•	Autotuner uses deterministic replay for cost models.
	•	Ensures bitwise reproducibility across runs and hardware.

⸻

5.9 Summary

	•	Tessera replaces CUDA’s grid/block/thread with IR-driven scheduling.
	•	Execution is controlled by Graph IR → Schedule IR → Tile IR → Target IR.
	•	Pipelines, fusion, and streams enable high utilization and low latency.
	•	Determinism and reproducibility are first-class properties.
