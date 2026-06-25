# Runtime execution matrix

**Generated from `tessera.compiler.execution_matrix._MATRIX` — do not hand-edit.**
Regenerate with:

```
python3 -c 'from tessera.compiler.execution_matrix import write_dashboard; write_dashboard()'
```

Single source of truth for what `runtime.launch()` does with each `(target, compiler_path)` pair. `capabilities.py`, `runtime.launch()`, and this dashboard all derive from the same `_MATRIX`. The drift test `test_runtime_execution_matrix` fails if they diverge.

## Executable rows

| Target | Compiler path | Executor | Execution kind | Telemetry mode | Reason |
|--------|---------------|----------|----------------|----------------|--------|
| `apple_cpu` | `apple_cpu_accelerate` | `apple_cpu_accelerate` | `native_cpu` | `cpu_accelerate` | Apple CPU artifact runs through Accelerate cblas_sgemm + multi-op chain. |
| `apple_cpu` | `apple_value_target_ir` | `apple_value_target_ir` | `native_cpu` | `cpu_accelerate` | Apple CPU value-call (tessera_apple.cpu.call) dispatches to the named Accelerate/LAPACK C ABI symbol. |
| `apple_gpu` | `apple_gpu_mps` | `apple_gpu_mps` | `native_gpu` | `metal_runtime` | Apple GPU artifact runs through MPS / MSL / MPSGraph per the runtime envelope. |
| `apple_gpu` | `apple_value_target_ir` | `apple_gpu_value_target_ir` | `native_gpu` | `metal_runtime` | Apple GPU value-call (tessera_apple.gpu.kernel_call) dispatches named C ABI symbols for strict rank-3 batched matmul, native sparse attention, PPO policy-loss, and EBM value envelopes. |
| `cpu` | `jit_cpu_numpy` | `jit_cpu_numpy` | `reference_cpu` | - | CPU JIT artifact runs through the numpy reference path. |
| `cpu` | `native_cpu` | `native_cpu` | `native_cpu` | - | CPU artifact runs through the x86 AMX / native CPU runtime. |
| `nvidia_sm120` | `nvidia_mma` | `nvidia_mma` | `native_gpu` | `cuda_runtime` | NVIDIA sm_120 matmul via the shipped warp-level mma.sync GEMM (tessera_nvidia_mma_gemm_{f16,bf16,tf32} C ABI symbol in libtessera_nvidia_gemm.so, NVRTC-compiled for the device arch; f16/bf16/fp32(tf32-math) storage, f32 accumulate). Directly selectable by stamping compiler_path="nvidia_mma". |
| `rocm` | `rocm_compiled` | `rocm_compiled` | `native_gpu` | `hip_runtime` | ROCm matmul artifact runs the COMPILER-GENERATED RDNA WMMA GEMM (Stage L): tessera-opt generates + serializes the kernel to hsaco in-process (no mlir-opt), then HIP loads + launches it. The DEFAULT rocm matmul lane; degrades to the hand-written rocm_wmma oracle when the compiled lane is unavailable on the host. |
| `rocm` | `rocm_flash_attn_compiled` | `rocm_flash_attn_compiled` | `native_gpu` | `hip_runtime` | ROCm flash_attn artifact runs the COMPILER-GENERATED RDNA WMMA FA-2 forward: tessera-opt generates + serializes the kernel to hsaco in-process, then HIP loads + launches it. The attention analog of the compiled GEMM lane (rocm_compiled). |
| `rocm` | `rocm_linear_attn_compiled` | `rocm_linear_attn_compiled` | `native_gpu` | `hip_runtime` | ROCm linear-attention-family artifact runs the COMPILER-GENERATED RDNA WMMA forward (quadratic-parallel form, no softmax): tessera-opt generates + serializes the kernel to hsaco in-process, then HIP loads + launches it. Handles linear_attn + lightning_attention (identity+decay) + retention (x²+decay) by op name. |
| `rocm` | `rocm_norm_compiled` | `rocm_norm_compiled` | `native_gpu` | `hip_runtime` | ROCm norm artifact runs the COMPILER-GENERATED RDNA row-reduction kernel (unweighted rmsnorm / layer_norm over the last axis, one workgroup per row, LDS tree-reduce of Σx and Σx²): tessera-opt generates + serializes the kernel to hsaco in-process, then HIP loads + launches it. Handles tessera.rmsnorm(_safe) + tessera.layer_norm by op name. |
| `rocm` | `rocm_softmax_compiled` | `rocm_softmax_compiled` | `native_gpu` | `hip_runtime` | ROCm softmax artifact runs the COMPILER-GENERATED RDNA row-reduction kernel (stable softmax over the last axis, one workgroup per row, LDS tree-reduce): tessera-opt generates + serializes the kernel to hsaco in-process, then HIP loads + launches it. The first non-matmul/non-WMMA compiled ROCm kernel. |
| `rocm` | `rocm_wmma` | `rocm_wmma` | `native_gpu` | `hip_runtime` | ROCm matmul via the hand-written RDNA WMMA GEMM (tessera_rocm_wmma_gemm_{f16,bf16} C ABI symbol, HIPRTC-compiled for the device arch). Now the reference ORACLE + availability fallback for the compiled lane (rocm_compiled) — still directly selectable by stamping compiler_path="rocm_wmma". |

## Targets with no executable row

These targets are recognized by the capability registry (so an artifact can carry them and lower correctly) but have no executable runtime row. `launch()` returns `runtime_status = "unimplemented"` when the target capability is present, or `"missing_backend"` otherwise — never silent success, never a fabricated output.

```
nvidia_sm80, nvidia_sm90, nvidia_sm100, rocm_gfx90a, rocm_gfx940, rocm_gfx942, rocm_gfx950, rocm_gfx1100, rocm_gfx1151, rocm_gfx1200
```

## Known executor IDs

| Executor ID | What it runs |
|-------------|--------------|
| `apple_cpu_accelerate` | Apple Silicon CPU via the Accelerate cblas_sgemm shim |
| `apple_gpu_mps` | Apple Silicon GPU via MPS / MSL / MPSGraph (per envelope) |
| `apple_gpu_value_target_ir` | Apple GPU value-call dispatch — invokes the C ABI symbol named in a tessera_apple.gpu.kernel_call value op (rank-3 batched matmul f32/f16/bf16; native sparse attention and PPO policy-loss variants plus EBM quadratic energy/Langevin value kernels when their Metal/MPSGraph executor probes are active) |
| `apple_value_target_ir` | Apple CPU value-call dispatch — invokes the C ABI symbol named in a tessera_apple.cpu.call value op (Value Target IR sprint; CPU cholesky executable) |
| `jit_cpu_numpy` | JIT CPU fallback via the numpy reference path |
| `native_cpu` | x86 AMX / native CPU runtime via the C runtime ABI |
| `nvidia_mma` | NVIDIA GPU (consumer Blackwell sm_120) warp-level mma.sync GEMM via the shipped libtessera_nvidia_gemm.so tessera_nvidia_mma_gemm_{f16,bf16,tf32} C ABI symbol (NVRTC-compiled for the device arch; f16/bf16/fp32(tf32-math) storage, f32 accumulate) |
| `rocm_compiled` | AMD GPU RDNA WMMA GEMM the Tessera compiler GENERATES (Stage L): tessera-opt generates + serializes the kernel to hsaco in-process (no mlir-opt), then HIP loads + launches it. Opt-in; f16 storage, f32 accum; the rocm_wmma lane stays the default + oracle |
| `rocm_flash_attn_compiled` | AMD GPU RDNA WMMA FA-2 forward the Tessera compiler GENERATES (generate-wmma-flash-attn-kernel -> ROCDL -> hsaco, in-process via tessera-opt), then HIP loads + launches it. f16/bf16 storage, f32 softmax + accumulate; the attention analog of rocm_compiled |
| `rocm_linear_attn_compiled` | AMD GPU RDNA WMMA linear-attention forward the Tessera compiler GENERATES (generate-wmma-linear-attn-kernel -> ROCDL -> hsaco, in-process via tessera-opt), then HIP loads + launches it. Quadratic-parallel form O = (φ(Q)φ(K)ᵀ ⊙ causal [⊙ λ^(i-j)]) @ V, NO softmax; f16/bf16 storage, f32 accumulate. Handles tessera.linear_attn + the decay-masked siblings tessera.lightning_attention (identity+decay) and tessera.retention (x²+decay) by op name |
| `rocm_norm_compiled` | AMD GPU RDNA row-reduction rmsnorm / layer_norm the Tessera compiler GENERATES (generate-rocm-norm-kernel -> ROCDL -> hsaco, in-process via tessera-opt), then HIP loads + launches it. Unweighted row normalize over the last axis (one workgroup per row, LDS tree-reduce of Σx and Σx²); handles tessera.rmsnorm(_safe) + tessera.layer_norm by op name. f32/f16/bf16 storage, f32 reduce |
| `rocm_softmax_compiled` | AMD GPU RDNA row-reduction softmax the Tessera compiler GENERATES (generate-rocm-softmax-kernel -> ROCDL -> hsaco, in-process via tessera-opt), then HIP loads + launches it. Stable softmax over the last axis (one workgroup per row, LDS tree-reduce); the first non-matmul/non-WMMA compiled ROCm kernel. f32/f16/bf16 storage, f32 reduce |
| `rocm_wmma` | AMD GPU RDNA WMMA matrix-core GEMM via the shipped libtessera_rocm_gemm.so tessera_rocm_wmma_gemm_{f16,bf16} C ABI symbol (HIPRTC-compiled for the device arch; f16/bf16 storage, f32 accumulate) |
