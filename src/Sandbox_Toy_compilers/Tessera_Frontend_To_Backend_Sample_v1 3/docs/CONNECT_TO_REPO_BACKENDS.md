# Connecting to your `src/compiler/codegen` backends

This sample emits a textual **Tessera Target-IR** file you can feed into your repo's pass pipeline.

## Option A — shell out to your tool

1) Build your repo tool (e.g., `tessera-opt` or `mlir-opt` with your Tessera dialect).
2) Export env vars when running the sample:

```
export TESSERA_RUN_PIPE=1
export TESSERA_TOOL=mlir-opt
export TESSERA_PIPELINE=--pass-pipeline='builtin.module(tessera-lower,convert-to-llvm)'
python -m tilec.driver examples/matmul.tss --backend tessera --out build/tessera_mm
```

An `*.after.mlir` artifact will be created in the output dir.

## Option B — import a Python wrapper

If your repo exposes Python bindings (pybind11/cffi), add a small wrapper module and replace
`tilec/backends/codegen_tessera.py:emit(...)` to call your binding directly.
