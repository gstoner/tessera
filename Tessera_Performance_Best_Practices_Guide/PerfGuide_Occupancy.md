# Tessera Performance Best Practices Guide
## Chapter 2: Occupancy and Parallelism

---

### 2.1 Overview

**Occupancy** measures the ratio of active threads/warps per Streaming Multiprocessor (SM) relative to hardware limits.  
In Tessera, occupancy is influenced by:

- **Tile shape** (`BM, BN, BK`) at the Tile IR level
- **Register usage** per thread
- **Shared memory usage** per tile
- **Fusion depth** (Graph IR → Schedule IR lowering)

High occupancy is generally desirable, but **maximum occupancy ≠ maximum performance**. The optimal point balances parallelism, memory pressure, and compute utilization.

---

### 2.2 Occupancy Tradeoffs

1. **High Occupancy**
   - Pros: Better latency hiding, more warps ready to execute.
   - Cons: Increased register pressure, possible spilling to local memory.

2. **Low Occupancy**
   - Pros: Larger tile sizes per warp, fewer synchronization points, better data reuse.
   - Cons: Less ability to hide memory latency.

---

### 2.3 Tile Shape Selection

Tile shape (`BM, BN, BK`) determines how work is partitioned:

- **BM × BN** = output tile size (rows × cols)
- **BK** = inner reduction tile size (depth)

**Guidelines:**
- Choose `BM, BN` to maximize TensorCore efficiency (multiples of 16 or 32).
- Keep `BK` large enough to amortize memory loads but small enough to fit in registers.
- Balance across GPU generations (e.g., Hopper vs. GB200).

**Tessera Example (MatMul kernel):**
```python
from tessera import schedule, op

@op.kernel
def matmul(A, B, C):
    schedule.tile((BM, BN, BK), warp=(16, 16))
    schedule.use_tensor_cores(True)
    schedule.occupancy_target(min_warps=8, max_warps=16)
    return A @ B + C

2.4 Warp- and Thread-Level Parallelism

Tessera’s Tile IR exposes warp-level intrinsics for reductions, shuffles, and TensorCore ops.
	•	Warp-level parallelism: Good for reductions, softmax, FlashAttention.
	•	Thread-level parallelism: Good for elementwise ops, fine-grained fusions.

Best Practice: Use warp-level operations for communication-heavy kernels; thread-level for compute-heavy, embarrassingly parallel workloads.

⸻

2.5 Fusion Depth and Occupancy

Fusion can reduce memory traffic but increases register usage.

Tessera’s Schedule IR autotuner balances fusion depth vs occupancy automatically:

	•	Shallow fusion: Better occupancy, more global memory traffic.
	•	Deep fusion: Fewer loads/stores, but risk of register spilling.

Tip: Enable autotuner’s occupancy-aware cost model when fusing.

⸻

2.6 Worked Example: FlashAttention Occupancy Tuning

FlashAttention is sensitive to occupancy:

	1.	Tile Sizes
	•	Small BM improves parallelism but increases softmax synchronization.
	•	Large BK improves data reuse but can cause register spill.
	2.	Best Practice

```python
schedule.tile((128, 128, 64))
schedule.pipeline_stages(3)
schedule.async_copy(True)
schedule.occupancy_target(min_warps=12, max_warps=16)
```
Outcome

	•	Achieves ~90% TensorCore utilization
	•	Keeps occupancy high enough to hide HBM latency
	•	Avoids register spills by bounding fusion depth

2.7 Key Takeaways

	•	Occupancy is not the end goal; balanced utilization is.
	•	Tile shape and register pressure dominate occupancy decisions.
	•	Warp-level ops are essential for reductions and collective patterns.
	•	Use Tessera’s autotuner to explore occupancy/fusion tradeoffs automatically.

⸻



