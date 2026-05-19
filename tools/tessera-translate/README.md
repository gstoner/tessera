# tessera-translate

Inter-IR text translation CLI for Tessera.

## Status (2026-05-18)

**Python scaffold landed.**  The C++ MLIR translation tool (the MLIR
`mlir-translate`-style binary that converts between MLIR text and
external IR formats like StableHLO bytecode, GGUF metadata, or
SafeTensors headers) is gated on `tessera-opt` building against
MLIR 21.

The Python module `python/tessera/cli/translate.py` ships the
inter-IR translation surface that doesn't need the MLIR build:
StableHLO/GGUF/SafeTensors export glue from `tessera.aot` (S14),
exposed as a console script `tessera-translate` so callers have a
stable entry point even before the C++ side lands.

## What lives here

- `README.md` — this file.

## What lives in `python/tessera/cli/translate.py`

- `main(argv)` — CLI entry point.
- Subcommands:
  - `tessera-translate stablehlo --in <aot.zip> --out <stablehlo.mlir>`
  - `tessera-translate gguf --in <aot.zip> --out <model.gguf>`
  - `tessera-translate safetensors --in <aot.zip> --out <model.st>`

Each subcommand dispatches through `tessera.aot.{stablehlo_export,
gguf_export, safetensors_export}` (S14).

## Future C++ tool

When `tessera-opt` builds, this directory will hold:

- `tessera-translate.cpp` — the MLIR binary entry point.
- `CMakeLists.txt` — links MLIRIR + Tessera dialects.
- `test/` — lit fixtures.

The Python CLI and the C++ binary share the entry name
`tessera-translate`; the Python wheel installs the console script,
and the C++ build (when it lands) installs the binary under a
distinguished name (e.g., `tessera-translate-mlir`) to avoid the
collision.
