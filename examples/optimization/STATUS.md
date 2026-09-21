# Status: `scaffold`

Tracked by `python/tessera/compiler/examples_manifest.py`.

This directory is a **standalone sketch collection**, not a canonical Tessera
compiler example today.

## Why

The directory contains four standalone CPU programs, three CUDA sketches, and
one generic MLIR file. The CPU programs have CMake targets, but none of the
sources enters Tessera through `@tessera.jit`, Graph IR, the canonical driver,
or a Tessera backend. The SM90 WGMMA file explicitly performs a no-op, so the
directory cannot honestly serve as compiler or device evidence.

There is no canonical Tessera entry-point script for the manifest to execute;
the audit row therefore uses `README.md` as the nominal entry point with status
`scaffold`.

## Path forward

Choose one of two explicit outcomes:

1. Add a Tessera-native example that compiles a real operation, queries or
   selects a schedule, validates numerics, and reports evidence without mixing
   timing domains; then give that entry point its own manifest row.
2. Move these generic teaching sketches to the archive if they are no longer
   part of the project story.

Until that work lands, this scaffold ships unchanged.
