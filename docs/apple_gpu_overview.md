# Apple GPU backend — Phase 8 overview

> **Status:** Phase 8.3 → 8.4.7 complete. The apple_gpu runtime path executes single-op and multi-op chains for matmul, rope, softmax, gelu, flash_attn, rmsnorm, plus four fused kernels covering attention and MLP blocks. f32 is fully wired; f16/bf16 are wired for matmul, rope, softmax, gelu, flash_attn, and matmul→softmax / flash_attn fusion.

This document is the canonical reference for how the apple_gpu backend hangs together — what each kernel is, when each fusion fires, where the cross-layer hooks live, and how to add a new kernel or fusion.

---

## Big picture

```
@jit(target="apple_gpu")
        │
        ▼
Graph IR  ──►  Schedule IR  ──►  Tile IR  ──►  Target IR (apple_gpu)
                                                  │
                                                  ▼
                              tessera-lower-to-apple_gpu-runtime
                              (or artifact-only path)
                                                  │
                                                  ▼
                              MetalDeviceContext + MSL kernel cache
                                                  │
                                                  ▼
                              MTLCommandBuffer + MTLComputeCommandEncoder
                                                  │
                                                  ▼
                                           on-device execution
```

Three layered decisions gate execution:

1. **Compile-time gate** (`driver._is_apple_gpu_mps_executable`) — does the program qualify for runtime execution at all? If yes, `execution_mode = "metal_runtime"`; if no, `metal_artifact`.
2. **Chain detection** (`driver._apple_gpu_chain_kind`) — for multi-op programs, which fusion pattern (if any) does the SSA chain match?
3. **Runtime dtype dispatch** (`runtime._apple_gpu_dispatch_*`) — at JIT call time, the input array dtype picks between f32 / f16 / bf16 ctypes wrappers, since `@jit` is type-polymorphic.

---

## Kernel inventory

See [`apple_gpu_kernel_inventory.md`](apple_gpu_kernel_inventory.md) for the full table. High-level summary:

**Single-op:**
- `mps_matmul` (rank-2, MPS-backed) — Phase 8.3 (f32) + 8.4.4 (f16/bf16)
- `rope` (rank-2, MSL) — Phase 8.4 (f32) + 8.4.4.1 (f16/bf16)
- `softmax` (rank-2, axis=-1, MSL) — Phase 8.4.2 (f32) + 8.4.4.1 (f16/bf16)
- `gelu` (rank-2, MSL, tanh-approx) — Phase 8.4.2 (f32) + 8.4.4.1 (f16/bf16)
- `flash_attn` (rank-3, MSL, online softmax) — Phase 8.4.1 (f32) + 8.4.4.2 (f16/bf16)

**Fused 2-op:**
- `matmul_softmax` — Phase 8.4.3 (f32) + 8.4.4.2 (f16/bf16). Plus tiled f32 variant (Phase 8.4.6) lifting N≤256 to N≤8192.
- `matmul_gelu` — Phase 8.4.7 (f32)
- `matmul_rmsnorm[_safe]` — Phase 8.4.7 (f32)

**Fused 3-op:**
- `matmul_softmax_matmul` — Phase 8.4.5 (f32/f16/bf16). Full attention block; `O = softmax(A@B) @ C`.

---

## Pipeline ordering

`tessera-lower-to-apple_gpu-runtime` runs passes in this order (longest fusion first wins greedy pattern matching):

```
1. matmul_softmax_matmul fusion   (3 ops, benefit=3)
2. matmul_softmax       fusion   (2 ops, benefit=2)
3. matmul_gelu          fusion   (2 ops, benefit=2)
4. matmul_rmsnorm       fusion   (2 ops, benefit=2)
5. matmul (mps)         lowering (single op)
6. rope                 lowering
7. flash_attn           lowering
8. softmax              lowering
9. gelu                 lowering
```

**Why ordering matters:** the rewrites use `RewritePattern` benefits, but pass-level ordering is the primary control. A `matmul → softmax → matmul` chain would otherwise get caught by the 2-op `matmul → softmax` pass first, losing the 3-op opportunity.

---

## Anatomy of a kernel emission

Every apple_gpu kernel has 4 cross-layer artifacts:

### 1. MSL source string + cache key
Lives in `python/tessera/compiler/target_ir.py` as a top-level constant:

```python
_APPLE_GPU_FOO_MSL_SOURCE = (
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void foo_f32(...)\n"
    "{ ... }\n"
)
_APPLE_GPU_FOO_MSL_CACHE_KEY = _sha256_short(_APPLE_GPU_FOO_MSL_SOURCE)
```

The cache key is a sha256 prefix — the runtime keys its `MTLComputePipelineState` cache by `(msl_source, entry_point)`.

### 2. C ABI symbol in `apple_gpu_runtime.mm`
The runtime shim has one function per (kernel × dtype):

```c++
extern "C" void tessera_apple_gpu_foo_f32(
    const float* A, ..., float* O, int32_t M, ...);
```

The Darwin path goes Metal/MPS; non-Darwin builds get a portable C++ reference fallback in `apple_gpu_runtime_stub.cpp`.

### 3. Lowering pass in `lib/Target/Apple/Lowering/`
A `RewritePattern` that matches the Tessera Graph IR op (or chain) and emits a `func.call` into the runtime symbol. Standard plumbing:

```cpp
extractPtr  → bufferization.to_buffer + memref.extract_aligned_pointer_as_index + arith.index_cast (i64)
allocate output memref + extract its pointer
emit constants for shape (i32) + scale/eps (f32) if needed
ensureExternalDecl(symbol, FunctionType{...})
emit func::CallOp
emit bufferization.to_tensor for the result
rewriter.replaceOp / eraseOp
```

### 4. Python dispatcher + ctypes wrapper in `runtime.py`
At launch time:

```python
def _apple_gpu_dispatch_foo(operands, np):
    # gate on dtype/shape
    if not _eligible(...):
        return _numpy_reference(...)
    sym = _apple_gpu_foo_f32()  # ctypes wrapper, lazy-loaded
    sym(...)
    return out

def _apple_gpu_foo_f32():
    runtime = _load_apple_gpu_runtime()
    sym = runtime.tessera_apple_gpu_foo_f32
    sym.argtypes = [...]
    sym.restype = None
    return sym
```

The dispatcher is invoked by `_execute_apple_gpu_mps_metadata` based on the metadata `op_name` (single-op) or chain detection (`_apple_gpu_metadata_is_*_chain`).

---

## How to add a new kernel

For a single-op kernel (e.g., `tessera.silu`):

1. **MSL source + C symbol** in `apple_gpu_runtime.mm` + stub fallback
2. **Lowering pass** in `lib/Target/Apple/Lowering/SiluToAppleGPU.cpp`
3. **Pass header declaration** in `Passes.h`
4. **Wire into pipeline** in `Passes.cpp` (in the per-op section, after fusions)
5. **CMake source** entry in `CMakeLists.txt`
6. **MSL source constant + cache key** in `target_ir.py`; extend `_apple_gpu_kernel_msl_for_dtype` if templating dtype variants
7. **Single-op envelope entry** in `target_ir.py::_apple_gpu_module_is_mps_runtime` and `target_ir.py::_lower_apple_gpu_op`
8. **Driver gating** — extend `driver.py::_APPLE_GPU_MSL_OPS`, backend artifact symbol selection
9. **Python dispatcher + ctypes wrapper** in `runtime.py`; add to envelope + dispatcher table; loader gate update
10. **Tests**: lit fixture + unit tests (artifact contract + end-to-end + ABI shim)

## How to add a new fusion

For a 2-op or 3-op fusion (e.g., `matmul → silu`):

1. **Fused MSL kernel + C symbol** in `apple_gpu_runtime.mm`
2. **Lowering pass** matching the chain at the consumer, walking up to verify `hasOneUse()` on intermediates
3. **Wire into pipeline before per-op passes** (longer chains before shorter)
4. **Driver chain detection** — extend `_apple_gpu_chain_kind` to recognize the new pattern
5. **Target IR fusion-kind detection** — extend `_apple_gpu_module_fusion_kind` to classify by source-set + op order; extend the per-pass MSL source/cache_key dispatch
6. **Runtime metadata-chain detector** — extend `_apple_gpu_metadata_is_*_chain` family, add dispatcher entry
7. **`_apple_gpu_module_is_mps_runtime`** — accept the new chain shape (remember to enforce chain ORDER, not just shape)
8. **Tests** following the existing pattern

---

## Compile-time vs runtime dispatch (important nuance)

`@jit(target="apple_gpu")` is **type-polymorphic** — the decorator doesn't see call-site dtypes. This means:

- The static **Graph IR** assumes `f32` operand types unless explicit type hints are present
- The compile-time **backend artifact** names `tessera_apple_gpu_*_f32` symbols by default
- At **launch time**, the `_apple_gpu_dispatch_*` helpers inspect the `numpy.dtype` of the actual input arrays and route to the matching ctypes wrapper

So the f16/bf16 paths exist *both* at compile time (when explicit type hints are used in pass-level lit fixtures) and at runtime (when the user passes f16/bf16 numpy arrays). The two paths share the same ABI; they just decide *when* to pick the symbol.

This split is why:
- **Lit fixtures** test compile-time dtype selection with explicit `tensor<*xf16>` types
- **Python unit tests** test runtime dtype dispatch with `np.float16` / `ml_dtypes.bfloat16` arrays

---

## MetalDeviceContext + kernel cache

```cpp
struct MetalDeviceContext {
  id<MTLDevice>       device;
  id<MTLCommandQueue> queue;
  bool                ok;
  std::unordered_map<std::string, id<MTLComputePipelineState>> kernel_cache;
  std::mutex          kernel_cache_mu;
};
```

- **Process-wide singleton** initialized lazily on first call.
- **Cache key** = `msl_source + '\x1f' + entry_point` — same source + same entry point hits the cache on subsequent calls.
- **Re-check under lock** in `compile_msl_kernel` to handle the race where two threads compile the same kernel concurrently.

The cache means the ~1ms MSL compile cost is paid once per unique kernel per process. Subsequent dispatches are O(buffer setup + encode + commit + wait).

---

## Constraints summary

- **Per-thread stack-array kernels** (everything except `mps_matmul`): N ≤ 256 (or D ≤ 256 for flash_attn).
- **Threadgroup-tiled matmul_softmax** (Phase 8.4.6 f32 only): N ≤ 8192.
- **Static shapes only**. Dynamic shape paths fall back to artifact-only / numpy reference.
- **Single-use intermediates** in fusion patterns. Multi-use intermediates fall back to per-op execution.
- **Matching dtypes within a chain.** Mixed-dtype chains fall back to per-op.

---

## Followups (deliberately deferred)

- **More dtype variants** for tiled matmul_softmax (f16/bf16 tiled), MLP fusions (f16/bf16), 3-op fusion threadgroup tiling
- **MPSGraph baseline benchmarks** for matmul vs MPS — needs a separate dependency
- **Chain extension** — `matmul → softmax → matmul → gelu`, `softmax → matmul → matmul`, etc.
- **Conv2D / pooling** kernels — the apple_gpu envelope is currently dense-matmul-shaped
- **Multi-batch matmul** — matmul currently rank-2; rank-3 batched would mirror the CPU path

---

## Files at a glance

| Concern | Path |
|---------|------|
| Runtime shim (Darwin) | `src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm` |
| Runtime stub (non-Darwin) | `src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime_stub.cpp` |
| Lowering passes | `src/compiler/codegen/Tessera_Apple_Backend/lib/Target/Apple/Lowering/` |
| Pipeline registration | `src/compiler/codegen/Tessera_Apple_Backend/lib/Target/Apple/Passes.cpp` |
| Pass declarations | `src/compiler/codegen/Tessera_Apple_Backend/include/Tessera/Target/Apple/Passes.h` |
| MSL source constants + helper | `python/tessera/compiler/target_ir.py` |
| Compile-time gating | `python/tessera/compiler/driver.py` |
| Runtime dispatcher + ctypes | `python/tessera/runtime.py` |
| Benchmark harness | `benchmarks/apple_gpu/benchmark_fusion.py` |
| Unit tests | `tests/unit/test_apple_backend_roadmap.py` |
| Lit fixtures | `tests/tessera-ir/phase8/apple_gpu_*.mlir` |
