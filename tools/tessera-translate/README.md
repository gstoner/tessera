# tessera-translate

Two complementary translation surfaces:

1. **Python `tessera-translate`** (console script, installed by
   `pyproject.toml`).  Wraps `tessera.aot` exports for AOT artifact
   → external format conversions (StableHLO text, GGUF binary,
   SafeTensors).  No MLIR linkage required.

2. **C++ `tessera-translate-mlir`** binary.  MLIR-native translation
   driver over the Tessera dialects + LLVM IR / SPIR-V.  Built via
   `cmake --build build --target tessera-translate-mlir`.

## Status (2026-05-18 — both shipped)

| Surface                    | Entry                               | Status   |
|----------------------------|-------------------------------------|----------|
| Python format export       | `tessera-translate <subcommand>`    | shipped  |
| C++ MLIR translate         | `tessera-translate-mlir`            | shipped  |
| Python `mlir` pass-through | `tessera-translate mlir <flags>`    | shipped  |

## Usage

```sh
# AOT artifact → external format (Python).
tessera-translate stablehlo  --in artifact.zip --out model.mlir
tessera-translate gguf       --in artifact.zip --out model.gguf
tessera-translate safetensors --in artifact.zip --out model.st
tessera-translate info       --in artifact.zip

# MLIR translation (C++ binary, called directly).
tessera-translate-mlir --mlir-to-llvmir input.mlir   > output.ll
tessera-translate-mlir --import-llvm    input.ll     > output.mlir
tessera-translate-mlir --serialize-spirv input.mlir  > output.spv

# Or via the Python pass-through (auto-detects the C++ binary
# under build/tools/tessera-translate/ or on PATH).
tessera-translate mlir --mlir-to-llvmir input.mlir > output.ll
```

## Files

- `tessera-translate.cpp` — C++ MLIR translation driver source.
- `CMakeLists.txt` — Builds the `tessera-translate-mlir` target.
- `README.md` — this file.

## What the C++ binary registers

- All Tessera dialects (`tessera`, `tessera.neighbors`,
  `tessera.solver`, `tessera_apple`, `tpp`) — conditional on the
  build options that pull each dialect in.
- Standard MLIR dialects needed for translation: arith / func /
  linalg / memref / scf / tensor / bufferization / LLVM / NVVM / ROCDL.
- Standard MLIR translations: `--mlir-to-llvmir` /
  `--import-llvm` plus the LLVM-IR dialect translation patterns
  (builtin / LLVM / NVVM / ROCDL).

## What the Python CLI registers

- 4 subcommands: `stablehlo`, `gguf`, `safetensors`, `info` (all
  pass through `tessera.aot.{stablehlo_export, gguf_export,
  safetensors_export}` from S14).
- 1 pass-through subcommand: `mlir <args...>` invokes the C++
  binary; reports a clean diagnostic if the binary isn't built.
