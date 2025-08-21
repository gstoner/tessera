# Tessera Interop & Tooling Guide
## Chapter 5: Profiling & Autotuning

---

### 5.1 Overview

Performance optimization in Tessera is driven by **profiling** and **autotuning**:

- **Profiler**: captures runtime metrics (latency, FLOPs, memory, bandwidth).  
- **Autotuner**: explores tiling, fusion, and pipeline schedules.  
- **Persistent caches**: tuned configs stored per shape & architecture.  

---

### 5.2 Runtime Profiler

Enable profiling in Python:

```python
from tessera import profiler, op

with profiler.session() as p:
    X = op.tensor((1024, 1024))
    Y = op.softmax(op.matmul(X, X))
    Y.eval()

p.report()
```
Sample output:
|Op          |Latency(ms)   |FLOPs(G)   |Bandwidth(GB/s)   |Efficiency(%)|
|------------|--------------|-----------|------------------|-------------|
|matmul      |3.21          |214.5      |1510.2            |92|
|softmax     |0.67          |2.1        |987.0             |88|

5.3 MLIR Cost Models

Schedule IR integrates with cost models:
```mlir 
%1 = "tessera.schedule.tile"(%0)
       {tile_sizes = [128, 128, 32], cost_model = "roofline"}
```
	•	Roofline model: estimates FLOPs vs bandwidth.
	•	Hardware-aware models: tuned for SM count, HBM, NVLink.

5.4 Autotuning Workflow

Autotuning tries multiple configurations:

```python
from tessera import autotune, op

cfg = autotune(op.matmul, shapes=(1024, 1024, 1024))
print(cfg)
```
Output:
```
Best config: tile=(128,128,64), pipeline_depth=3, fusion=True
```
5.5 Persistent Caches

Autotuned results are stored in shape-arch caches:
	•	Cache key: (op, shape, dtype, arch)
	•	Cache storage: $HOME/.tessera/autotune/

```python
cfg = autotune.load(op.matmul, (1024, 1024, 1024))
```
Caches enable fast startup without re-tuning.

⸻

5.6 On-Device Measurements

Tessera autotuner supports online measurements:

```python
cfg = autotune(op.matmul,
               shapes=(8192,8192,8192),
               method="on_device")

	•	Runs candidate kernels on GPU.
	•	Selects best performer.
	•	Stores results in persistent cache.

⸻

5.7 Advanced Profiling

Profiler supports:
	•	Timeline traces (Chrome trace format).
	•	Hardware counters (SM occupancy, warp divergence).
	•	Memory breakdown (HBM, L2, shared).

```python
    profiler.timeline("trace.json")
```
5.8 Summary
	•	Tessera profiler gives detailed runtime metrics.
	•	Autotuner explores schedule space with cost models.
	•	Persistent caches accelerate repeated workloads.
	•	On-device measurements ensure performance portability.
