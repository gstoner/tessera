# Tessera ROCm Backend

This backend defines ROCm hardware-free Target IR contracts and optional ROCm
tooling. The active Python artifact path lowers through the verified object
compiler spine:

```text
textual DSL / @jit -> Graph IR -> Schedule IR -> Tile IR -> ROCm Target IR
```

`python/tessera/compiler/target_ir.py` maps Tile IR to:

- `tessera_rocm.mfma` for matmul/MMA contracts
- `tessera_rocm.async_copy` and `tessera_rocm.wait` for LDS movement contracts
- `tessera_rocm.elementwise` for generic elementwise artifacts
- `tessera.target.diagnostic` for unsupported contracts such as KV-cache and
  FlashAttention in this phase

These artifacts are covered by `tests/unit/test_target_ir.py` and
`tests/unit/test_target_ir_contract.py`. Native HIP/HSACO execution is separate
from artifact generation and should only be claimed for flows that use the
runtime/tooling below.

## Build
```bash
mkdir build && cd build
cmake -G Ninja -DMLIR_DIR=<path>/lib/cmake/mlir -DLLVM_DIR=<path>/lib/cmake/llvm \
      -DTESSERA_ROCM_BUILD_TOOLS=ON -DTESSERA_ROCM_BUILD_RUNTIME=ON ..
ninja
```

## Lower + Emit
```bash
# Lowering tests
ninja check-tessera-rocm

# Emit HSACO (auto-detects toolchain; try --mcpu=gfx90a|gfx94|gfx1200)
./tessera-rocm-emit ../test/rocm/async_and_mfma_realish.mlir out.hsaco --mcpu=gfx90a
# Writes out.hsaco and out.hsaco.metadata.json
```

## Launch (HIP)
```bash
./launch_demo out.hsaco <kernel_symbol>
```

## CK Bridge
Enable with `-DTESSERA_ROCM_ENABLE_CK=ON`. If `composable_kernels` is discoverable, the bridge uses it; else it logs a stub.
