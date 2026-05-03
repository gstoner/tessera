#!/usr/bin/env bash
# CPU-only validation spine for Tessera.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  PYTHON="${PYTHON_FALLBACK:-python3}"
fi

TMP_ROOT="${TMPDIR:-/tmp}"
RUNTIME_BUILD="$TMP_ROOT/tessera_runtime_validate"
TPROF_BUILD="$TMP_ROOT/tessera_tprof_validate"
BENCH_REPORT="$TMP_ROOT/tessera_benchmark_smoke.json"
RUNTIME_REPORT="$TMP_ROOT/tessera_runtime_smoke.json"

cd "$ROOT"

echo "==> Version consistency"
"$PYTHON" scripts/check_versions.py

echo "==> Python unit tests"
"$PYTHON" -m pytest tests/unit -q

echo "==> Runtime telemetry smoke"
"$PYTHON" -m tessera.cli.runtime --output "$RUNTIME_REPORT"
"$PYTHON" benchmarks/perf_gate.py "$RUNTIME_REPORT" --baseline benchmarks/baselines/cpu_smoke.json

echo "==> Benchmark telemetry smoke"
"$PYTHON" benchmarks/run_all.py --smoke --json-only --output "$BENCH_REPORT"
"$PYTHON" benchmarks/perf_gate.py "$BENCH_REPORT" --baseline benchmarks/baselines/cpu_smoke.json

echo "==> Standalone CPU runtime build and tests"
cmake -S src/runtime -B "$RUNTIME_BUILD" -DCMAKE_BUILD_TYPE=Release -DTESSERA_BUILD_TESTS=ON
cmake --build "$RUNTIME_BUILD" -j
ctest --test-dir "$RUNTIME_BUILD" --output-on-failure

echo "==> C++ profiler build and smoke"
cmake -S tools/profiler -B "$TPROF_BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$TPROF_BUILD" -j
"$TPROF_BUILD/tprof" \
  --demo-out "$TMP_ROOT/tprof_demo.trace.json" \
  --perfetto-out "$TMP_ROOT/tprof_demo.perfetto.json" \
  --report-out "$TMP_ROOT/tprof_demo.report.html" \
  --peaks tools/profiler/scripts/peaks_sample.yaml \
  --arch sm90

echo "==> Collectives runtime compile check"
c++ -std=c++17 -Isrc/collectives/include \
  -c src/collectives/lib/Dialect/Collective/Runtime/Execution.cpp \
  -o "$TMP_ROOT/tessera_collective_execution.o"

if [ -d build ]; then
  echo "==> Existing build tree ctest"
  ctest --test-dir build --output-on-failure
else
  echo "==> No repo build/ directory; skipping optional monorepo ctest"
fi

echo "Tessera CPU validation spine passed"
