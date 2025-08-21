# Tessera Interop & Tooling Guide
## Chapter 1: Python & C++ APIs

---

### 1.1 Overview

Tessera provides **two primary front-ends**:

- **Python API**  
  - High-level modeling and operator graph construction.  
  - Autodiff, scheduling, and distributed tensor abstractions.  
  - Ideal for ML researchers and model prototyping.

- **C++ API**  
  - Low-level runtime and ABI interface.  
  - Integration into production systems and custom kernels.  
  - Access to memory management, scheduling, and execution primitives.

Both APIs interoperate seamlessly. Python builds operator graphs, while C++ executes optimized IR pipelines.

---

### 1.2 Python API Basics

The Python API centers around **operator graphs**:

```python
from tessera import op, graph, dist

# Define a simple MLP
X = op.tensor((B, D), dtype="bf16")
Y = op.mlp(in_dim=D, hidden=[8192], out_dim=D)(X)

# Training step with autodiff
@graph.training_step
def step(batch):
    out = Y(batch["input"])
    loss = op.cross_entropy(out, batch["labels"])
    grads = graph.backward(loss)
    return grads, {"loss": loss}
```
	•	op.tensor creates symbolic tensors.
	•	Operators (MLP, matmul, softmax) compose into graphs.
	•	Autodiff integrates directly with the graph IR.

⸻
1.3 Distributed Tensors in Python

Tessera supports distributed tensors with explicit sharding:
```python
mesh = dist.Mesh(axes=["dp","tp"], devices=range(8))

W = dist.tensor((1024, 1024),
                layout=dist.ShardSpec(partition=("row","col"), mesh_axes=("dp","tp")),
                mesh=mesh)

Y = op.matmul(W, W.T)
```
- Sharding specifications are part of the IR.
- Tessera lowers to NVLink/InfiniBand collectives automatically.

1.4 C++ API Basics

For integration into systems, the C++ runtime API mirrors Python:

```cpp
#include <tessera/runtime.h>
using namespace tessera;

int main() {
    // Create a tensor
    Tensor A = Tensor::create({1024, 1024}, DType::BF16);

    // Define an operator (MatMul)
    Tensor B = Tensor::create({1024, 1024}, DType::BF16);
    Tensor C = op::matmul(A, B);

    // Execute
    Runtime rt;
    rt.submit(C);
    rt.wait();

    return 0;
}
```

- C++ exposes tensor creation, operator invocation, and runtime submission.
- Compatible with CUDA streams and ROCm HIP execution contexts.

1.5 Python ↔ C++ Interop

- Python graphs can be serialized to IR and executed in C++.
- C++ runtime can register custom ops that are visible in Python.

Example: Exporting graph from Python to C++:
```python
compiled = graph.compile(Y, target="ptx")
compiled.save("mlp.tsr")
```
```cpp
Graph g = Graph::load("mlp.tsr");
Runtime rt;
rt.submit(g);
```
1.6 Use Cases

	•	Python API → Model research, operator prototyping.
	•	C++ API → Production deployment, embedded runtime.
	•	Combined → Prototype in Python, deploy in C++.

⸻

1.7 Summary

	•	Tessera supports Python for modeling and C++ for systems integration.
	•	Distributed tensors and operator graphs are available in both.
	•	Seamless serialization enables moving between front-ends.

⸻
