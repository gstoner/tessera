# Tessera Interop & Tooling Guide
## Appendix A: Reference Commands & Examples

---

### A.1 Python CLI Tools

Run a Tessera program:

```bash
tessera run my_model.py
```
Dump intermediate representations:

```bash
tessera-mlir my_model.py --emit=graph-ir
tessera-mlir my_model.py --emit=schedule-ir
tessera-mlir my_model.py --emit=tile-ir
```
Profile a program:

```bash
tessera-prof my_model.py --trace trace.json
```
Autotune a kernel:

```bash
tessera-tune my_model.py --op=matmul --shape=8192,8192,8192
```

A.2 C++ Interop Snippets

Minimal Tessera C++ API usage:
```cpp
#include <tessera/tessera.h>

using namespace tessera;

int main() {
    Tensor A = Tensor::rand({1024, 1024}, DType::BF16);
    Tensor B = Tensor::rand({1024, 1024}, DType::BF16);

    Tensor C = op::matmul(A, B);

    C.eval();
    return 0;
}
```
Embed Tessera in a C++ runtime:

```cpp
Graph g;
auto X = g.add_input({32, 1024}, DType::F32);
auto Y = op::softmax(op::matmul(X, X));

g.compile();
g.run();
```

A.3 MLIR Dialect Cheatsheet

Graph IR

```mlir
%0 = "tessera.graph.matmul"(%A, %B)
```
Schedule IR

```mlir
%1 = "tessera.schedule.tile"(%0)
       {tile_sizes = [128, 128, 32]}
```

Tile IR

```mlir
%2 = "tessera.tile.mma_sync"(%1)
       {warp_size = 32}
```

Target IR (LLVM/PTX)

```mlir
%3 = llvm.call @llvm.nvvm.wmma.m16n16k16.mma.sync(...)
```
A.4 Debugging Commands

Dump IR with debugging info:

```bash
tessera-mlir my_model.py --emit=graph-ir --debug
```

Check gradients:

```python
from tessera.debug import check_grad
check_grad(model.loss, wrt=model.params)
```

Enable determinism check:

```python
from tessera.debug import check_determinism
check_determinism(model.loss)
```

A.5 Profiling Commands

Run profiler:

```bash
tessera-prof my_model.py --metrics=flops,bandwidth,occupancy
```

Generate Chrome trace:

```bash
tessera-prof my_model.py --trace=trace.json
chrome://tracing  # open in browser
```
A.6 Summary

The appendix consolidates:

	•	CLI tools for running, profiling, autotuning.
	•	C++ snippets for embedding Tessera.
	•	MLIR cheatsheet for Graph → Schedule → Tile → Target IR.
	•	Debugging & profiling commands for performance analysis.

⸻
