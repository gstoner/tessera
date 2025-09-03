# Tessera_ROCM_Backend_Starter_v3

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
