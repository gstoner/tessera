# Tessera /p3d/ package

This drop‑in package provides the **P3D** dialect (3D conv/pyramid/global context),
pass stubs, tests, docs, and a sample benchmark.

## Build (standalone)
```bash
cmake -S . -B build && cmake --build build -j
ctest --test-dir build
```

## Use with tessera-opt
Link `TesseraP3DInit.o` into `tessera-opt` or compile alongside your tree so the
passes `-tessera-lower-p3d` and `-tessera-autotune-p3d` are registered.
