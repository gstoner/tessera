#!/usr/bin/env bash
set -euo pipefail
OUTDIR=${1:-bench/reports}
mkdir -p "${OUTDIR}"
BIN=./bench_gpu
CSV="${OUTDIR}/gpu_results.csv"
HTML="${OUTDIR}/gpu_report.html"
if [ ! -x "${BIN}" ]; then
  echo "bench_gpu not found; build with -DTESSERA_BUILD_GPU_BENCH=ON" 1>&2
  exit 1
fi
"${BIN}" > "${CSV}"
python3 ../scripts/gpu_report.py --csv "${CSV}" --out "${HTML}"
echo "CSV : ${CSV}"
echo "HTML: ${HTML}"
