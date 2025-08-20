
# Tessera Performance Best Practices Guide
## Chapter 7: Distributed Execution

---

### 7.1 Overview

Tessera provides **first-class distributed execution** via **mesh abstractions** and **shard specifications**.  
These enable models to scale from **single GPU** → **multi-GPU node** → **NVLink/NVSwitch clusters** with minimal code changes.

Key concepts:
- **Mesh**: logical topology of devices (1D, 2D, 3D).
- **Shards**: how tensors are partitioned across mesh axes.
- **Collectives**: communication primitives lowered to NVLink/NVSwitch/NCCL.
- **Parallelism modes**: data parallel, tensor parallel, pipeline parallel, expert parallel.

---

### 7.2 Mesh Abstraction

Define a device mesh once, reuse across operators:

```python
from tessera import dist

# Create 2D mesh over 16 GPUs (4×4)
mesh = dist.Mesh(axes=["tp", "dp"], devices=range(16))

print(mesh.topology)
# tp × dp mesh, 4 × 4

```

### 7.3 Sharding Tensors

Tensors can be explicitly sharded across mesh axes:7.3 Sharding Tensors

Tensors can be explicitly sharded across mesh axes:

```python
W = dist.tensor(shape=(1024, 1024),
                layout=dist.ShardSpec(partition=("row", "col"),
                                      mesh_axes=("tp", "dp")),
                mesh=mesh, dtype="bf16")
```


This shards W by rows across tensor-parallel axis, and by columns across data-parallel axis.

⸻

7.4 Collectives

Tessera lowers collective operations to the appropriate backend:
	•	All-reduce → NVLink/NCCL ring or tree algorithm.
	•	Reduce-scatter / All-gather → used in tensor parallel matmuls.
	•	Pipeline send/recv → optimized NVSwitch point-to-point transfers.

Example:
```python
from tessera import op

Y_local = op.matmul(W_local, X_local)
Y_global = dist.all_reduce(Y_local, axis="dp")   # combine across data-parallel axis
```
7.5 Parallelism Strategies
	1.	Data Parallel (DP)
	•	Replicate model across mesh axis.
	•	Gradients reduced via all-reduce.
	•	Best for small models with large batch sizes.
	2.	Tensor Parallel (TP)
	•	Split weight matrices across GPUs.
	•	Requires reduce-scatter/all-gather around matmuls.
	•	Essential for trillion-parameter models.
	3.	Pipeline Parallel (PP)
	•	Split layers across GPUs.
	•	Microbatch scheduling overlaps forward/backward passes.
	•	Reduced activation memory per device.
	4.	Expert Parallel (EP)
	•	MoE layers distribute experts across GPUs.
	•	Router decides which expert to activate.
	•	Communication minimized with hierarchical meshes.

⸻

7.6 Example: Hybrid Parallel Training

```python
# Define 3D mesh: tensor × pipeline × data parallel
mesh = dist.Mesh(axes=["tp","pp","dp"], devices=range(72))

# Shard transformer weights across tp × pp
W = dist.tensor((100_000, 100_000),
                layout=dist.ShardSpec(partition=("row","col"),
                                      mesh_axes=("tp","pp")),
                mesh=mesh)

# Training step with hybrid parallelism
@graph.training_step(module="Transformer")
def step(batch):
    out = transformer(batch["input"], W)
    loss = op.cross_entropy(out, batch["labels"])
    grads = graph.backward(loss)
    return grads, {"loss": loss}
```
This runs across:
	•	Tensor-parallel groups for sharded matmuls.
	•	Pipeline groups for sequential layers.
	•	Data-parallel groups for scaling batches.

⸻

7.7 Overlapping Communication & Compute

Tessera’s Schedule IR allows explicit overlap:

```python
schedule.all_reduce_overlap(True)   # gradient sync overlaps with compute
schedule.pipeline_microbatches(8)   # interleave forward/backward
```
This hides communication latency and maximizes GPU utilization.

7.8 Deterministic Execution

Distributed training can diverge without control of reduction order.
Tessera ensures determinism by:
	•	Fixed reduction order across collectives.
	•	Numerics policy applied globally (deterministic vs fast).
	•	Reproducible training across identical cluster configs.

⸻

7.9 Best Practices
	•	Use 3D meshes (tp × pp × dp) for trillion-parameter models.
	•	Keep microbatch count ≥ pipeline depth for overlap.
	•	Enable overlap of comm/compute to minimize idle time.
	•	Use hierarchical collectives (intra-node, inter-node) for multi-rack scaling.

⸻

7.10 Key Takeaways
	•	Tessera abstracts distributed execution into mesh + shard specs.
	•	Parallelism modes (DP, TP, PP, EP) compose seamlessly.
	•	Autotuner + overlap features maximize performance on NVLink/NVSwitch clusters.
	•	Deterministic reductions ensure reproducibility at scale.
