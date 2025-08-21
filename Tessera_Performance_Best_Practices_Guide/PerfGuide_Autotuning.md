# Tessera Performance Best Practices Guide
## Chapter 4: Autotuning Strategies

---

### 4.1 Overview

Tessera includes a **built-in autotuner** that automatically explores tile shapes, pipeline depths, vectorization widths, and fusion depths.  
It combines:

- **Analytical cost models** → estimate FLOP/byte ratio, occupancy, memory bandwidth.  
- **On-device measurements** → actual kernel timings, capturing hardware effects.  
- **Persistent caches** → store best schedules per (operator, shape, architecture).  

Autotuning is critical because *optimal parameters vary with GPU architecture, tensor shape, and precision*.

---

### 4.2 Workflow

1. **Candidate Generation**  
   - Tessera generates a search space of possible schedules (tiling, unrolling, async copy usage).  

2. **Cost Model Filtering**  
   - Analytical model eliminates obviously poor candidates.  
   - Prioritizes high arithmetic intensity, coalesced access, valid occupancy.  

3. **On-Device Benchmarking**  
   - Selected candidates are JIT-compiled and timed on target hardware.  
   - Measurements include latency, throughput, achieved occupancy, bandwidth utilization.  

4. **Cache Storage**  
   - Best-performing schedule is stored in a **persistent cache** (per-shape + per-arch).  
   - Cache is re-used for future runs → avoids repeated tuning.  

---

### 4.3 Autotuner API

```python
from tessera import autotune, op

# Enable autotuning globally
autotune.enable(True)

# Example: autotuning a matmul kernel
@op.kernel
def matmul(A, B, C):
    return A @ B + C

# Run with autotuner
Y = matmul(X, W)

# Export tuned schedules for reuse
autotune.save_cache("nvlink72_cache.json")

# Import cache (e.g., on another machine with same arch)
autotune.load_cache("nvlink72_cache.json")
```
4.4 Best Practices
	•	Warmup Phase
	•	Allow autotuner to run for first few iterations of training.
	•	Later iterations reuse tuned schedules → no runtime penalty.
	•	Cache Management
	•	Export/import caches between runs and clusters.
	•	Use arch-specific cache names (sm90_cache.json, gb200_cache.json).
	•	Shape Generalization
	•	Tessera can interpolate schedules across nearby shapes.
	•	Best for convolution-like ops where shapes vary slightly.
	•	Avoid Over-Tuning
	•	Excessively large search spaces waste tuning time.
	•	Use cost model constraints to prune invalid candidates.

⸻

4.5 Example: Depthwise Convolution Autotuning

Depthwise convolution is memory-bound and sensitive to tile shape.
Tessera autotuner explores:
	•	Tile sizes (BM, BN) for input feature maps.
	•	Vectorization factors (x1, x2, x4).
	•	Shared memory buffering strategies.

Schedule IR Example:

```python
schedule.search_space(
    tile_sizes=[(32, 32), (64, 64), (128, 64)],
    vectorize=[1, 2, 4],
    pipeline_stages=[2, 3]
)

schedule.autotune(
    cost_model="roofline",
    max_trials=50
)
`
Outcome:
	•	Autotuner discovers (64,64) tiling with vectorize=4 and 3-stage pipeline.
	•	Achieves ~90% bandwidth utilization on GB200.
	•	Reduces training time by ~20% compared to hand-picked schedule.

⸻

4.6 Autotuning with Fusion
	•	Fusion depth is included in the autotuning search.
	•	Autotuner balances lower memory traffic (deep fusion) vs register pressure.
	•	Cost model tracks register usage → prunes fusions that cause spills.

⸻

4.7 Key Takeaways
	•	Use the autotuner for all performance-critical ops.
	•	Persist caches for repeatability and multi-run efficiency.
	•	Cost model + measurement → faster convergence to best schedule.
	•	Fusion depth and tiling are highly hardware-dependent → trust the tuner.