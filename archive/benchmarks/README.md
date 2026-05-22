# Archived Benchmarks

This directory keeps benchmark experiments that are useful historical context
but are not active compiler-backed benchmarks.

`matrix_multiplication/` is an older Blackwell concept sketch. It used
high-level APIs that are not part of the current Tessera compiler/runtime
surface, so future Blackwell work should land as Target IR tests, runtime
kernels, or operator benchmark cases instead of reviving it as-is.
