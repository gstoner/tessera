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
| `apple_gpu` | `apple_value_target_ir` | `apple_gpu_value_target_ir` | `native_gpu` | `metal_runtime` | Apple GPU value-call (tessera_apple.gpu.kernel_call) dispatches named C ABI symbols for strict rank-3 batched matmul and native sparse attention envelopes. |
| `cpu` | `jit_cpu_numpy` | `jit_cpu_numpy` | `reference_cpu` | - | CPU JIT artifact runs through the numpy reference path. |
| `cpu` | `native_cpu` | `native_cpu` | `native_cpu` | - | CPU artifact runs through the x86 AMX / native CPU runtime. |

## Targets with no executable row

These targets are recognized by the capability registry (so an artifact can carry them and lower correctly) but have no executable runtime row. `launch()` returns `runtime_status = "unimplemented"` when the target capability is present, or `"missing_backend"` otherwise — never silent success, never a fabricated output.

```
nvidia_sm80, nvidia_sm90, nvidia_sm100, nvidia_sm120, rocm, rocm_gfx90a, rocm_gfx940, rocm_gfx942, rocm_gfx950, rocm_gfx1100, rocm_gfx1200, metalium
```

## Known executor IDs

| Executor ID | What it runs |
|-------------|--------------|
| `apple_cpu_accelerate` | Apple Silicon CPU via the Accelerate cblas_sgemm shim |
| `apple_gpu_mps` | Apple Silicon GPU via MPS / MSL / MPSGraph (per envelope) |
| `apple_gpu_value_target_ir` | Apple GPU value-call dispatch — invokes the C ABI symbol named in a tessera_apple.gpu.kernel_call value op (rank-3 batched matmul f32/f16/bf16; native sparse attention and PPO policy-loss variants when their Metal/MPSGraph executor probes are active) |
| `apple_value_target_ir` | Apple CPU value-call dispatch — invokes the C ABI symbol named in a tessera_apple.cpu.call value op (Value Target IR sprint; CPU cholesky executable) |
| `jit_cpu_numpy` | JIT CPU fallback via the numpy reference path |
| `native_cpu` | x86 AMX / native CPU runtime via the C runtime ABI |
