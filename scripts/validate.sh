#!/usr/bin/env bash
# CPU-only validation spine for Tessera.
#
# Step ordering follows `docs/spec/VALIDATION_SPINE.md` (M5).  The
# Python-side gates (unit, audit drift, claim_lint, canonical-program
# reports, benchmark-schema validation) run first; native build /
# benchmark checks run after.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$HOME/venv/bin/python" ]; then
    PYTHON="$HOME/venv/bin/python"
  elif [ -x "$ROOT/.venv/bin/python" ]; then
    PYTHON="$ROOT/.venv/bin/python"
  else
    PYTHON="${PYTHON_FALLBACK:-python3}"
  fi
fi

TMP_ROOT="${TMPDIR:-/tmp}"
RUNTIME_BUILD="$TMP_ROOT/tessera_runtime_validate"
TPROF_BUILD="$TMP_ROOT/tessera_tprof_validate"
BENCH_REPORT="$TMP_ROOT/tessera_benchmark_smoke.json"
RUNTIME_REPORT="$TMP_ROOT/tessera_runtime_smoke.json"

cd "$ROOT"

echo "==> Environment bootstrap"
"$PYTHON" scripts/validation_env.py

echo "==> Version consistency"
"$PYTHON" scripts/check_versions.py

echo "==> Python lint (ruff)"
# ruff catches unused imports, undefined names, common bugs, security
# smells (S301/S403 — pickle.loads), and a curated subset of pyupgrade
# rules.  Config in pyproject.toml; baseline is "zero errors" — the
# whole package is currently clean.
if "$PYTHON" -m ruff --version >/dev/null 2>&1; then
  "$PYTHON" -m ruff check python/tessera/
else
  echo "warning: ruff not installed in this Python (skipping lint)"
fi

echo "==> Python type check (mypy ratchet)"
# mypy runs against the baseline tracked in scripts/mypy_baseline.txt.
# Pass when the error count is at or below the baseline; fail loudly
# when a new error lands.  After a focused cleanup sprint update the
# baseline via `scripts/mypy_ratchet.sh --update-baseline`.
# Derive the mypy binary that lives next to the Python interpreter.
PYTHON_BIN_DIR="$(dirname "$PYTHON")"
if [ -x "$PYTHON_BIN_DIR/mypy" ]; then
  MYPY="$PYTHON_BIN_DIR/mypy" scripts/mypy_ratchet.sh
elif "$PYTHON" -m mypy --version >/dev/null 2>&1; then
  # Fall back to invoking via `python -m mypy` for installs without
  # the console script.
  MYPY="$PYTHON -m mypy" scripts/mypy_ratchet.sh
else
  echo "warning: mypy not installed in this Python; skipping ratchet"
fi

echo "==> Python unit tests"
"$PYTHON" -m pytest tests/unit -q -m "not slow"

# M0 / M0.5: regenerate the support table and fail if it drifts from
# the checked-in copy.  Catches changes to op_catalog / primitive
# coverage / backend_manifest / capabilities that didn't update the
# audit artifact.
echo "==> Generated support-table drift check"
"$PYTHON" -m tessera.compiler.audit support_table --check

# M0 follow-up: claim_lint — public docs may not assert native /
# fused / hardware-runtime claims that the manifest can't ground.
echo "==> Public-doc claim_lint"
"$PYTHON" -m tessera.compiler.audit claim_lint

echo "==> Runtime telemetry smoke"
"$PYTHON" -m tessera.cli.runtime --output "$RUNTIME_REPORT"
"$PYTHON" benchmarks/perf_gate.py "$RUNTIME_REPORT" --baseline benchmarks/baselines/cpu_smoke.json

echo "==> Benchmark telemetry smoke"
"$PYTHON" benchmarks/run_all.py --smoke --json-only --output "$BENCH_REPORT"
"$PYTHON" benchmarks/perf_gate.py "$BENCH_REPORT" --baseline benchmarks/baselines/cpu_smoke.json

# Apple-Silicon-aware GA/EBM native-execution health check.
# Promoted to the validation spine 2026-05-17 — see
# docs/status/ga_ebm_milestone.md for the contract.  On non-Darwin
# hosts the benchmark exits 0 with `skipped_apple_gpu` populated.
# On Apple Silicon it exercises 17 GA + 9 native EBM + 4 workload
# rows through the full stack (Python API → manifest lookup → MSL
# dispatch → correctness check vs Python reference).  Counts
# updated 2026-05-17 after `ebm_partition_exact` shipped.
GA_EBM_REPORT="$TMP_ROOT/tessera_ga_ebm_native_smoke.json"
echo "==> Apple GPU GA + EBM native-execution smoke"
"$PYTHON" benchmarks/apple_gpu/benchmark_ga_ebm.py --ci --output "$GA_EBM_REPORT"

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

# Opt-in C++ sanitizer smoke.  Off by default because each sanitizer
# build is ~2 minutes (Debug + sanitizer flags + collectives toolchain).
# Enable in CI lanes that care about runtime-safety regressions:
#   TESSERA_VALIDATE_SANITIZERS=1 scripts/validate.sh
# Run a subset via:
#   TESSERA_VALIDATE_SANITIZERS=asan scripts/validate.sh
if [ "${TESSERA_VALIDATE_SANITIZERS:-0}" != "0" ]; then
  case "${TESSERA_VALIDATE_SANITIZERS}" in
    1|all|true) SANS="asan tsan ubsan" ;;
    *)          SANS="${TESSERA_VALIDATE_SANITIZERS}" ;;
  esac
  echo "==> Sanitizer smoke ($SANS)"
  scripts/run_sanitizers.sh $SANS
fi

echo "Tessera CPU validation spine passed"
