# Tessera API Reference - Volume 4
## Runtime and Deployment

### 1. Runtime Architecture
- MLIR-based compilation pipeline【24†source】
- Host-device orchestration【22†source】

---

### 2. Host Integration

CUDA launch example【23†source】:
```cpp
extern "C" cudaError_t launch_tessera_flash_attention(
    const half* Q, const half* K, const half* V, half* O,
    float scale, int batch_size, int num_heads, int seq_len, int head_dim,
    cudaStream_t stream = nullptr
);
```

---

### 3. Autotuning and Profiling

Autotuning search space【22†source】:
```mlir
tessera.autotuning = {
  tile_sizes = [[64,64],[128,128],[256,64]],
  objectives = {minimize="latency", maximize="occupancy"}
}
```

Profiling【23†source】:
```cpp
std::cout << "Runtime: " << profile_data.average_runtime_ms << " ms
";
std::cout << "Tensor Core Utilization: " << profile_data.tensor_core_utilization << "%
";
```

---

### 4. Production Deployment

#### C API【23†source】
```c
TesseraCompilationResult tessera_target_compile_module(
  TesseraTargetCompiler* compiler,
  const char* mlir_module_text,
  const char* target_platform, const char* target_architecture, int opt_level);
```

#### CMake Integration【23†source】
```cmake
set(CMAKE_CUDA_ARCHITECTURES 90)
target_link_libraries(flash_attention_example CUDA::cudart CUDA::cublas)
```

---

### 5. Example Models

#### Transformer with FlashAttention【21†source】
```python
class TransformerBlock(tessera.Module):
    def __init__(self, dim, heads):
        self.attn = tessera.nn.FlashAttention(dim, heads)
        self.ffn = tessera.nn.MLP(dim, 4*dim)
```

#### MLA Transformer【17†source】
```python
layer = MLATransformerLayer(model_dim=4096, latent_dim=512, mlp_dim=14336)
```

#### Quantized SwiGLU【19†source】
```python
@tessera.quantized(weights=int8, activations=fp8)
def swiglu_quantized(x, W_gate, W_up, W_down, scales): ...
```
