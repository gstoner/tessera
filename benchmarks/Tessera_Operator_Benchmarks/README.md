# Tessera Operator Benchmarks (opbench)

This package provides a portable, backend-agnostic operator micro-benchmark
suite for the Tessera Programming Model and current compiler artifact contract.

It focuses on *operator-level* kernels with controlled inputs, sweep configs, CSV logging, and HTML reports.

**Key goals**
- Comparable numbers across CPU/GPU/accelerators via a unified harness.
- Clean separation between CPU reference timing, Tessera compiler artifacts, and
  future Tessera runtime execution.
- Reproducible sweeps with YAML configs and seeded RNG.
- Reference implementations for correctness checks.
- Shared `tessera.telemetry.v1` events in JSON rows and sweep summaries.

**Directory layout**
```
common/          # timers, NVTX helpers, device init hooks
harness/         # CLI runner, op registry, config parser
ops/             # individual operators (matmul, conv2d, attention, …)
mlir/            # Tessera IR samples for each op
scripts/         # python sweeps, Tessera bridge, CSV/HTML report tools
docs/            # spec split into two parts (merge markers included)
.github/workflows# CI example
```

**Quick start**
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/opbench --list-ops
./build/opbench --op matmul --m 64 --n 64 --k 64 --iters 3 --seed 123 --json
~/venv/bin/python scripts/opbench.py --config scripts/configs/quick_sweep.yaml --bin ./build/opbench --out /tmp/opbench_out
~/venv/bin/python scripts/opbench.py --config scripts/configs/quick_sweep.yaml --bin ./build/opbench --backend artifact --out /tmp/opbench_artifacts
~/venv/bin/python scripts/opbench.py --config scripts/configs/quick_sweep.yaml --bin ./build/opbench --backend tessera-runtime --runtime bridge --out /tmp/opbench_runtime
~/venv/bin/python scripts/report_html.py /tmp/opbench_out/results.csv /tmp/opbench_out/report.html
```

Backends:
- `reference`: executable CPU reference kernels with correctness checks.
- `artifact`: generates a Tessera JIT bundle for each operator and validates
  Graph, Schedule, Tile, and Target artifacts; reports
  `runtime_status="artifact_only"` and `telemetry.status="unmeasured"`.
- `tessera-runtime --runtime bridge`: launches generated CPU `RuntimeArtifact`
  bundles through the Python `@tessera.jit` + `ts.launch()` path for all
  registered operators.
- `tessera-runtime --runtime native`: reserved for the future generated C ABI
  dispatch layer and reports `backend_unavailable` until that path is built.
- If ROCm/CUDA are available, define `-DOPBENCH_WITH_NVTX=ON` to get NVTX ranges.

Current operator coverage:

| Operator | CPU reference | Generated artifact bundle | Tessera runtime bridge | Native C ABI |
|---|---:|---:|---:|---:|
| `matmul` | yes | Graph/Schedule/Tile/Target | executable | pending |
| `conv2d` | yes | Graph/Schedule/Tile/Target | executable | pending |
| `flash_attention` | yes | Graph/Schedule/Tile/Target | executable | pending |
| `reduce` | yes | Graph/Schedule/Tile/Target | executable | pending |
| `elementwise` | yes | Graph/Schedule/Tile/Target | executable | pending |
| `softmax_layernorm` | yes | Graph/Schedule/Tile/Target | executable | pending |
| `transpose_gather` | yes | Graph/Schedule/Tile/Target | executable | pending |

Known gaps:
- Native generated C ABI launch hooks remain pending; use `--runtime bridge`
  for the portable generated-artifact execution path.
- Static MLIR samples remain readable examples, but benchmark artifact
  validation now uses generated bundles.
- CSV output keeps nested telemetry as JSON-like strings for spreadsheet
  compatibility; use `results.json` for structured telemetry analysis.

**Merging the docs**
The spec is split into two markdowns. Merge between markers:
```
--- BEGIN-MERGE: Operator_Benchmarks_Spec ---
... (Part 1) ...
--- END-MERGE: Operator_Benchmarks_Spec ---
```
and in Part 2 the same markers bracket the continuation.
