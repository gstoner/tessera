# Status: `runnable` kernel benchmark; agent orchestrator scaffold

Tracked by `python/tessera/compiler/examples_manifest.py`.

The manifest runs `examples/kernel_autotuning/benchmark_kernel.py`. It loads a
candidate tile configuration, validates the tiled implementation against a
reference, and fails on invalid parameters or numerical mismatch.

## Agent boundary

`src/agents/tree_search_runner.py` orchestrates an end-to-end LLM +
tree-search agent (sketch of the system from arXiv:2509.06503,
*"An AI system to help scientists write expert-level empirical
software"*). To run end-to-end it needs:

* A real LLM client (the bundled `DummyLLM` only proposes literal
  `print('hello from variant {i}')` stubs).
* A sandbox executor that can actually run candidate code outside
  the orchestrator's process.
* A per-task harness with scorable inputs (the `examples/` subfolders
  currently ship README shells only, except for `kernel_autotuning/`).

None of those pieces are present in CI, so the orchestrator is
intentionally not run as a smoke test.

The orchestrator itself still needs a real LLM client and sandbox before it can
be promoted. Its `DummyLLM` output is not used as manifest evidence.
