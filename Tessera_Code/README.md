# Tessera Starter Source (Compiler + Runtime + Python Frontend)

This is a **scaffold** for Tessera's compiler stack (MLIR-based), runtime, and Python DSL.
It is intended as a starting point you can flesh out into a full implementation.

## Layout
```
compiler/              # MLIR dialects + passes + lowering stubs
runtime/               # Runtime ABI, device/stream/memory + backends (CUDA/ROCm stubs)
python/                # Python DSL front-end + ctypes binding hooks
tools/                 # Autotuner placeholder
examples/              # Tiny examples
.github/workflows/     # CI stubs
```

## Build (runtime only, portable stub)
```bash
cmake -S . -B build -DTESSERA_ENABLE_CUDA=OFF -DTESSERA_ENABLE_ROCM=OFF
cmake --build build -j
```
This builds a stub `libtessera_runtime` with no GPU dependencies.

## Optional: Enable CUDA / ROCm backends (stubs)
```bash
cmake -S . -B build -DTESSERA_ENABLE_CUDA=ON   # requires CUDA toolchain
cmake -S . -B build -DTESSERA_ENABLE_ROCM=ON   # requires ROCm toolchain
```

## Python (local editable)
```bash
pip install -e ./python
python examples/hello_tessera.py
```

## Next Steps
- Flesh out MLIR dialects in `compiler/mlir/dialects/*.td` and register passes.
- Implement real backends in `runtime/src/backends/` and collective adapters.
- Wire Python ops to compiler-generated kernels (currently ctypes stubs).
