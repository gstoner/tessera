# Tessera Interop & Tooling Guide
## Chapter 4: Debugging Tools

---

### 4.1 Overview

Debugging large-scale ML workloads is challenging.  
Tessera provides a **layered debugging toolkit** for:

- **Graph inspection** (visualize operators, dependencies).  
- **Numerical tracing** (track intermediate tensor values).  
- **Autodiff validation** (gradient checking).  
- **Determinism checks** (bitwise reproducibility across devices).  

---

### 4.2 Graph Inspection

The **graph tracer** allows visualizing computation:

```python
from tessera import graph, op

X = op.tensor((32, 1024))
Y = op.softmax(op.matmul(X, X.T))

graph.trace(Y).print()
```
Example output:
```
%0 = tensor<32x1024xf32>
%1 = matmul %0, %0^T
%2 = softmax %1
```
	•	Supports IR dumps at Graph IR, Schedule IR, Tile IR levels.
	•	Export to MLIR or GraphViz for visualization.

⸻

4.3 Numerical Tracing

Debug values at runtime:
```python
with graph.debug_trace():
    out = Y.eval()
```
Produces logs of tensor values:
```
Tensor %1: mean=0.01, std=0.12
Tensor %2: max=0.99, min=0.00
```
Supports:

	•	Tensor summaries (mean, std, min, max).
	•	Sample values (configurable).

⸻

4.4 Gradient Checking

Validate autodiff with finite differences:
```python
from tessera.debug import check_grad

check_grad(Y, wrt=[X], eps=1e-4, atol=1e-3)
```
Output
```
Gradient check passed for %X (max error 2.1e-5)
```
4.5 Determinism Checks

Verify reproducibility across devices:
```python
from tessera.debug import check_determinism

check_determinism(Y, runs=5)
```
Output:
```
All 5 runs produced identical results (bitwise)
```

4.6 Integration with External Debuggers

	•	Python: integrates with pdb, IPython.
	•	C++: hooks for gdb, lldb.
	•	MLIR: supports mlir-opt --debug-only=tessera.

⸻

4.7 Summary

Tessera debugging tools provide:

	•	Graph-level inspection and visualization.
	•	Runtime numerical tracing.
	•	Autodiff validation via gradient checks.
	•	Determinism and reproducibility guarantees.