# Archived Benchmarks

This directory keeps benchmark experiments that are useful historical context
but are not active compiler-backed benchmarks.

`matrix_multiplication/` is an older Blackwell concept sketch. It used
high-level APIs that are not part of the current Tessera compiler/runtime
surface, so future Blackwell work should land as Target IR tests, runtime
kernels, or operator benchmark cases instead of reviving it as-is.

`tesserabench_docs/` (archived 2026-09-17) is the eight-document TesseraBench
design set that used to live at `docs/benchmarks/`. It describes a
`tesserabench` package, CLI and production stack that were never built — 23 of
its 26 named classes and 13 of its 14 module paths exist nowhere in the tree.
See its own README for the check; `docs/benchmarks/README.md` now indexes what
actually runs.

