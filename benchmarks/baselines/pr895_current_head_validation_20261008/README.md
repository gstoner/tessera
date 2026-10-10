# PR895 current-head contract validation

Owner E2E-REAL-6 / W1.1 / AD-RESIDUAL-EVAL-1.
Source head f8c571b88 is bound by receipt.json.

44 exact RTX 5070 cases pass nested NVFP4 leading maps and bounded
attention JIT/native AD with physical bias, compiler-free reuse and retained
outputs. GPU UUID, active SM12.0 and compiler/runtime hashes are recorded.
No fresh performance comparison or sibling-device evidence is inferred.

Ruff passes. The unchanged mypy ratchet passes errors=0 / baseline=0.
GitHub quality and required lint checks pass on this source head.
247 focused host checks pass with 23 skips; seven tests initially failed
because they monkeypatched an unused helper. The actual native exporter is
now patched instead, strengthening the before-compiler assertion. The complete
affected fixture file passes 15 cases. Both logs are retained.

All 32 generated documents are in sync and Graphify refresh completes.
Five broken relative evidence links are corrected; documentation lint passes.

Generic scaled_matmul batching/transpose closure and remaining aggregate CI
gates stay open. Coverage states, tests and CI baselines are unchanged.
