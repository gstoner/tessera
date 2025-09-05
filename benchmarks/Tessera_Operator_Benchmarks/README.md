# Tessera Operator Benchmarks (opbench)

This package provides a portable, backend-agnostic operator micro-benchmark suite for the Tessera Programming Model.

It focuses on *operator-level* kernels with controlled inputs, sweep configs, CSV logging, and HTML reports.

**Key goals**
- Comparable numbers across CPU/GPU/accelerators via a unified harness.
- Clean separation between *op definition* and *backend implementation* (Tessera Target IR / vendor libs).
- Reproducible sweeps with YAML configs and seeded RNG.
- Reference implementations for correctness checks.

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
./build/opbench --op matmul --m 2048 --n 2048 --k 2048 --iters 100 --seed 123
python3 scripts/opbench.py --config scripts/configs/quick_sweep.yaml --bin ./build/opbench --out /tmp/opbench_out
python3 scripts/report_html.py /tmp/opbench_out/results.csv /tmp/opbench_out/report.html
```

Backends:
- If Tessera runtime is available, define `-DOPBENCH_WITH_TESSERA=ON` and implement the hooks in each op.
- If ROCm/CUDA are available, you can optionally define `-DOPBENCH_WITH_NVTX=ON` to get NVTX ranges.

**Merging the docs**
The spec is split into two markdowns. Merge between markers:
```
--- BEGIN-MERGE: Operator_Benchmarks_Spec ---
... (Part 1) ...
--- END-MERGE: Operator_Benchmarks_Spec ---
```
and in Part 2 the same markers bracket the continuation.

