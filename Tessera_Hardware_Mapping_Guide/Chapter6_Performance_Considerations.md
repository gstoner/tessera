# Tessera Hardware Mapping Guide
## Chapter 6: Performance Considerations

Details on occupancy, latency hiding, Tensor Core utilization, autotuning strategies.

---

### 6.1 Overview

Performance in Tessera depends on efficiently mapping Graph IR, Schedule IR, and Tile IR to GPU hardware.  
This chapter details optimization strategies for:

- **Occupancy tuning**  
- **Latency hiding**  
- **Tensor Core utilization**  
- **Bandwidth balancing**  
- **Autotuning strategies**  

---

### 6.2 Occupancy Tuning

**Occupancy** = fraction of active warps per SM.  
Tessera’s autotuner dynamically balances tile sizes to maximize occupancy without register oversubscription.

Key factors:

- Tile size (M, N, K)  
- Shared memory usage  
- Register pressure  
- Warp count per block  

Example (Schedule IR autotuned tiling):
```mlir
%0 = "tessera.schedule.tile"(%A, %B)
       {m=128, n=128, k=64, autotune=true}
```
6.3 Latency Hiding

Tessera pipelines memory and compute:

	•	Prefetch next tile while computing current tile.
	•	Overlap communication (all-reduce) with local computation.

Execution pattern:
load (N+1), compute (N), comm (N-1)

Tessera runtime inserts async barriers to enforce correct overlap.

⸻

6.4 Tensor Core Utilization

Tensor Cores achieve peak throughput only if:

	•	Tile dimensions align with hardware MMA shapes (e.g., 16×16×16).
	•	Data is stored in fragment-friendly layouts.
	•	Precision is chosen for hardware (FP16, BF16, TF32).

Tile IR example:
```mlir
%fragC = "tessera.tile.mma_sync"(%fragA, %fragB, %fragC)
          {m=16, n=16, k=16, dtype="bf16"}
```
6.5 Bandwidth Balancing

Performance bottlenecks often shift between compute and memory.
Tessera autotuner uses cost models to balance:

	•	Compute-bound kernels → maximize Tensor Core throughput.
	•	Memory-bound kernels → reduce tile size, increase prefetch overlap.
	•	Comm-bound kernels → fuse collectives with compute stages.

⸻

6.6 Autotuning Strategies

Tessera autotuner integrates:

	1.	Analytic cost models (tile sizes, occupancy, bandwidth).
	2.	On-device measurements for ground truth performance.
	3.	Persistent caches per (shape, architecture).

Workflow:

	1.	First run: explore tiling space.
	2.	Record best config in cache.
	3.	Subsequent runs: reuse cached schedule.
 ```python
@op.autotune(cache="h100_configs")
def matmul_autotuned(A, B):
    return op.matmul(A, B)
```
6.7 Mixed Precision Optimization
	•	Use FP8/BF16/TF32 where possible.
	•	Accumulate in FP32 for stability.
	•	Tessera supports mixed precision at Graph IR type annotations.

Example:
```python
A = dist.tensor((B, D), dtype="fp8_e4m3")
B = dist.tensor((D, H), dtype="bf16")
C = op.matmul(A, B, accum_dtype="fp32")
```
6.8 Summary

	•	Tessera maximizes performance by coordinating occupancy, tiling, and precision.
	•	Autotuner ensures schedules are hardware-optimized and cached.
	•	Deterministic execution provides reproducible benchmarking.
	•	Strategies balance compute vs memory vs communication bottlenecks.

⸻
