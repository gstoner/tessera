# Status: `scaffold`

Tracked by `python/tessera/compiler/examples_manifest.py`.

This directory is a **research scaffold**, not a runnable example today.

## Why

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

## Path forward

* `examples/kernel_autotuning/benchmark_kernel.py` is closer to
  runnable (pure-Python tile-config scoring with a numpy-ish
  reference) and could graduate to its own `runnable` row
  independent of the agent loop.
* The orchestrator itself would need a real LLM client + sandbox
  before it can be promoted past `scaffold`.

Until that work lands, this scaffold ships unchanged.
