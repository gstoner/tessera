#!/usr/bin/env bash
# profile.sh — Phase 3 counter collection (plan §4, §7). SEPARATE from timing.
# Targeted metric set (NOT --set full, which inflated the paper's Table V). Emits
# counters.jsonl keyed by (kernel, dtype, N) so the cross-check in
# check_consistency.py can match event timings. Metric names are CANDIDATES —
# resolve them on-box first:
#   ncu --query-metrics | grep -E 'dram__cycles_active|lts__t_sector_hit'
#
# WSL2 prerequisite: GPU performance counters must be enabled on the Windows host
# (NVIDIA Control Panel -> Developer -> Manage GPU Performance Counters ->
# "Allow access to all users"), else ncu fails with ERR_NVGPUCTRPERM.
#
# Args: profile.sh [BENCH] [SIZES_CSV] [OUT] [ENABLE]
#   ENABLE is passed through to bench so a probe-rejected family is never launched
#   under the profiler either (fail-closed, plan §Phase 0).
set -euo pipefail

BENCH=${1:-./bench}
SIZES=${2:-512,1024,2048,4096,8192}
OUT=${3:-counters.jsonl}
ENABLE=${4:-ALL}

METRICS=$(cat <<'EOF'
gpu__time_duration.sum
dram__cycles_active.avg
dram__throughput.avg.pct_of_peak_sustained_elapsed
lts__t_sector_hit_rate.pct
l1tex__t_sector_hit_rate.pct
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio
smsp__thread_inst_executed_per_inst_executed.ratio
smsp__sass_branch_targets_threads_divergent.sum
smsp__inst_executed.sum
sm__warps_active.avg.pct_of_peak_sustained_active
EOF
)
METRIC_CSV=$(echo "$METRICS" | paste -sd, -)

# Verify counter access before doing real work (fail fast, honest error).
if ! ncu --query-metrics >/dev/null 2>&1; then
  echo "ERROR: ncu cannot read metrics (ERR_NVGPUCTRPERM?). Enable GPU perf" >&2
  echo "counters on the Windows host, then 'wsl --shutdown' and retry." >&2
  exit 1
fi

: > "$OUT"
FAIL=0
# One ncu invocation PER SIZE so the same-named kernel launched at each N is not
# collapsed — N comes from the loop, not from a fragile demangled-name parse.
for n in $(echo "$SIZES" | tr ',' ' '); do
  RAW="ncu_raw_${n}.csv"
  rm -f "$RAW"                       # never parse a previous run's file
  if ! ncu --metrics "$METRIC_CSV" --csv --target-processes all --log-file "$RAW" \
       "$BENCH" --sizes "$n" --iters 1 --warmup 0 --enable "$ENABLE" >/dev/null 2>&1; then
    echo "ERROR: ncu failed for N=$n (version-dependent metric? profiling denied?)" >&2
    FAIL=1; continue
  fi
  if [ ! -s "$RAW" ]; then
    echo "ERROR: ncu produced no output for N=$n" >&2; FAIL=1; continue
  fi
  python3 parse_ncu.py "$RAW" "$n" >> "$OUT" || { echo "parse failed N=$n" >&2; FAIL=1; }
done

if [ "$FAIL" != 0 ]; then
  echo "Phase 3 FAILED — counters.jsonl is incomplete; do NOT trust the counter" >&2
  echo "columns of REPORT.md for the missing sizes." >&2
  exit 1
fi
echo "wrote $OUT"
