
# Tessera PDLL Patterns

This folder contains PDLL source patterns mirroring the C++ canonicalizations.
You can compile these with `mlir-pdll` to native rewriters or to PDL bytecode.

Examples:
```bash
mlir-pdll --gen-rewriters   -I ${MLIR_SRC_DIR}/include   src/tessera/pdll/canon_cleanup.pdll   -o ${BUILD_DIR}/src/tessera/pdll/PDLLRewriters.inc
```
And then include `PDLLRewriters.inc` into a small pass that registers them.

Alternatively compile to PDL:
```bash
mlir-pdll   src/tessera/pdll/canon_cleanup.pdll   -o ${BUILD_DIR}/share/tessera/patterns/canon_cleanup.pdl.mlir
```
and apply with the PDL interpreter (out of scope in this drop).
