# Tessera Performance Best Practices Guide
## Appendix: Reference Tables & Defaults

---

### A.1 Recommended Tile Sizes

| Operation       | Tile Sizes (BM × BN × BK) | Notes |
|-----------------|---------------------------|-------|
| GEMM (BF16)     | 128 × 128 × 64           | Standard Hopper/GB200 |
| GEMM (FP8)      | 256 × 128 × 128          | FP8 TensorCores prefer larger tiles |
| Convolution     | 128 × 128 × 32           | Im2col lowering |
| Attention (Flash)| 128 × 64 × 64           | Warp-specialized |
| LayerNorm       | 256 threads/block        | One row per warp |

---

### A.2 Occupancy Guidelines

| Architecture | Max Warps/SM | Registers/Thread (ideal) | Shared Memory/Block |
|--------------|--------------|---------------------------|---------------------|
| Hopper (H100)| 64           | ≤ 64                      | ≤ 96 KB             |
| GB200        | 96           | ≤ 48                      | ≤ 128 KB            |

- **Rule of thumb**: ≥ 50% occupancy is enough if arithmetic intensity is high.  
- Avoid spilling registers into local memory.  

---

### A.3 Precision Defaults

| Mode     | Input Compute | Accumulation | Loss Scaling | Use Case |
|----------|---------------|--------------|--------------|----------|
| Safe     | FP32          | FP32         | None         | Debug, small models |
| Standard | BF16/FP16     | FP32         | Dynamic      | Training |
| Aggressive| FP8          | BF16         | Dynamic      | Trillion-param models |
| Deterministic | Any      | FP32         | None         | Reproducibility tests |

---

### A.4 Autotuning Defaults

| Component        | Default Setting               | Notes |
|------------------|-------------------------------|-------|
| Search strategy  | Bayesian + grid hybrid        | Uses cost model priors |
| Warmup samples   | 50                            | Reduced with persistent cache |
| Persistence      | Enabled per `(shape, arch)`   | Cache stored in `.tessera/cache/` |
| Cost model       | Latency × (1/throughput)      | Tunable per operator |

**Tip**: Always persist tuning results between runs for stable performance.  

---

### A.5 Communication Patterns

| Parallelism | Collective        | Pattern          | Notes |
|-------------|-------------------|------------------|-------|
| Data-parallel | All-reduce      | Ring or tree     | Gradient sync |
| Tensor-parallel | Reduce-scatter + All-gather | Hierarchical | Used in matmuls |
| Pipeline    | Send/Recv         | Point-to-point   | Activation transfer |
| Expert      | All-to-all        | Hierarchical shuffle | MoE router |

---

### A.6 Microbatch Recommendations

- Pipeline depth = `d` → microbatch count ≥ `d`  
- Recommended: **2× pipeline depth** for overlap.  
- Balance microbatch size vs. activation memory.  

---

### A.7 Numerics Policies

```python
from tessera import numerics
```

|numerics.policy("fast")         |# Maximum speed          |
|--------------------------------|-------------------------|
|numerics.policy("deterministic")|# Reproducible reductions| 
|numerics.policy("kahan_sum")    |# Compensated summation  |

|Policy       | Use Case                         |
|-------------|----------------------------------|
|fast         | Benchmarking, inference          |
|deterministic| Debugging, reproducible training |
|kahan_sum    | Sensitive PDEs, physics training |

A.8 Checklist for Performance Tuning

	•	Check occupancy (≥ 50%).
	•	Avoid register spills.
	•	Ensure memory coalescing.
	•	Enable async copy + multi-stage pipeline.
	•	Use autotuner with persistent cache.
	•	Shard tensors explicitly across mesh axes.
	•	Overlap communication with compute.
	•	Apply mixed precision (BF16/FP8) with FP32 accumulation.
	•	Select numerics policy (deterministic if reproducibility required).

⸻

A.9 Further Reading

	•	Tessera Programming Guide (core language and IR concepts)
	•	Runtime & ABI Spec (execution model and low-level ABI)
	•	Tessera IR Docs (Graph IR, Schedule IR, Tile IR, Target IR)
	•	NVIDIA CUDA Best Practices Guide for background on GPU tuning

