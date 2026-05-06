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
scripts/         # python sweeps + CSV/HTML report tools
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
~/venv/bin/python scripts/report_html.py /tmp/opbench_out/results.csv /tmp/opbench_out/report.html
```

Backends:
- `reference`: executable CPU reference kernels with correctness checks.
- `artifact`: verifies the registered MLIR sample artifact exists for each op;
  reports `runtime_status="artifact_only"` and `telemetry.status="unmeasured"`.
- `tessera-runtime`: explicitly reports `backend_unavailable` until generated
  operator runtime launches are wired to the Tessera C ABI.
- If ROCm/CUDA are available, define `-DOPBENCH_WITH_NVTX=ON` to get NVTX ranges.

Current operator coverage:

| Operator | CPU reference | Compiler artifact sample | Runtime status |
|---|---:|---:|---|
| `matmul` | yes | yes | reference executable; Tessera runtime pending |
| `conv2d` | yes | yes | reference executable; Tessera runtime pending |
| `flash_attention` | yes | yes | reference executable; Tessera runtime pending |
| `reduce` | yes | yes | reference executable; Tessera runtime pending |
| `elementwise` | yes | yes | reference executable; Tessera runtime pending |
| `softmax_layernorm` | yes | yes | reference executable; Tessera runtime pending |
| `transpose_gather` | yes | yes | reference executable; Tessera runtime pending |

Known gaps:
- No operator benchmark currently launches a generated Tessera runtime kernel.
- Artifact samples are Graph IR contracts, not hardware-validated Target IR.
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
