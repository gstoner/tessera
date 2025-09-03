<<<MERGE_START: Tessera_TPU_PJRT_Execution>>>
# PJRT Compile & Execute (TPU)

Build with `-DTESSERA_HAVE_PJRT_C_API=ON` and ensure `pjrt_c_api.h` is discoverable.

```bash
cmake -S . -B build -DTESSERA_HAVE_PJRT_C_API=ON -DMLIR_DIR=... -DSTABLEHLO_DIR=...
cmake --build build -j
PJRT_DEVICE=TPU ./build/pjrt_runner --program examples/matmul_128.mlir --format=stablehlo
```

**Note:** The starter compiles the program. For full execution, add device buffers using
the PJRT buffer APIs and call `PJRT_Executable_Execute`.

<<<MERGE_END: Tessera_TPU_PJRT_Execution>>>
