# Tessera Interop & Tooling Guide
## Chapter 2: MLIR Integration

---

### 2.1 Overview

Tessera is built on **MLIR (Multi-Level Intermediate Representation)**.  
Each IR level in Tessera corresponds to a dialect or transformation pipeline:

- **Graph IR** → Algebraic operator graphs (autodiff, high-level ops).  
- **Schedule IR** → Tiling, fusion, and pipeline transformations.  
- **Tile IR** → Block/warp/TensorCore mappings.  
- **Target IR** → LLVM, PTX, ROCm backends.

---

### 2.2 Tessera MLIR Dialects

- **`tessera.graph`**  
  - High-level operators (`matmul`, `fft`, `softmax`).  
  - Autodiff primitives (`grad`, `backward`).  

- **`tessera.schedule`**  
  - Loop nests, tiling, fusion.  
  - Pipeline stages for compute/memory overlap.  

- **`tessera.tile`**  
  - Explicit thread/block layout.  
  - TensorCore intrinsics (`mma.sync`, `ldmatrix`).  

- **`tessera.target`**  
  - Lowers to LLVM dialect.  
  - Emits PTX (NVIDIA) or HIP/ROCm (AMD).  

---

### 2.3 Example: Lowering a MatMul

#### Python Definition
```python
from tessera import op

A = op.tensor((M, K), dtype="bf16")
B = op.tensor((K, N), dtype="bf16")
C = op.matmul(A, B)
```
MLIR Graph IR

```mlir 
%0 = "tessera.graph.matmul"(%A, %B)
       : (tensor<MxKxbf16>, tensor<KxNxbf16>) -> tensor<MxNxbf16>
```
MLIR Schedule IR (tiled)

```mlir
%1 = "tessera.schedule.tile"(%0)
       {tile_sizes = [128, 128, 32]}
       : (tensor<MxNxbf16>) -> tensor<MxNxbf16>
```
MLIR Tile IR (GPU mapping)

```mlir
%2 = "tessera.tile.mma_sync"(%1)
       {warp_size = 32, tensor_core = true}
       : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
```
Target IR (PTX via LLVM)
```mlir
%3 = llvm.call @llvm.nvvm.wmma.m16n16k16.mma.sync(...)
```
2.4 Extending Tessera with MLIR Passes

Users can add custom optimization passes:

```bash
tessera-opt my_model.tsr.mlir \
  --tessera-schedule-fuse \
  --tessera-autotune \
  --convert-tessera-to-llvm
  ```
	•	tessera-schedule-fuse → merges operators for better locality.
	•	tessera-autotune → inserts cost model hints and kernel configs.
	•	convert-tessera-to-llvm → lowers to LLVM IR for codegen.

2.5 Custom Operators

Custom ops can be introduced via MLIR:

```mlir
%4 = "tessera.graph.my_custom_op"(%X, %Y)
       {alpha = 0.1}
       : (tensor<MxDxf32>, tensor<DxNxf32>) -> tensor<MxNxf32>

Then lowered via a user-defined pass into schedule/tile dialects.

2.6 Use Cases
	•	Research: Prototyping new operator transforms.
	•	Systems: Extending lowering for custom hardware.
	•	Debugging: Inspecting IR at multiple levels.

⸻

2.7 Summary
	•	Tessera leverages MLIR dialects for multi-level IR.
	•	Graph → Schedule → Tile → Target pipelines provide modular compilation.
	•	Users can inspect, extend, and optimize via MLIR passes.

⸻
