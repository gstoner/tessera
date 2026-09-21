# Status: `runnable`

Tracked by `python/tessera/compiler/examples_manifest.py`.

The manifest runs `examples/kernel_autotuning/benchmark_kernel.py`. It loads a
candidate tile configuration, validates the tiled implementation against a
reference, and fails on invalid parameters or numerical mismatch.

The incomplete LLM/tree-search orchestrator and task shells have moved to
`archive/examples/advanced/Tessera_Empirical_Software_Agent/`. The active
directory contains only the maintained benchmark and its documentation.
