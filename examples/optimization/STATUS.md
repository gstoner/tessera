# Status: `runnable`

Tracked by `python/tessera/compiler/examples_manifest.py`.

The maintained entry point is `tessera_schedule_example.py`, a canonical
Tessera scheduled-matmul example with a NumPy oracle and compiler-artifact
checks.

## Boundary

The four CPU programs, three CUDA sketches, and generic MLIR source remain
standalone teaching material. In particular, the SM90 WGMMA file still contains
a no-op and is not compiler or device evidence. The manifest executes only the
canonical Tessera entry point.
