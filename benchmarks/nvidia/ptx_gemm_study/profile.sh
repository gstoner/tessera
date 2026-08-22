#!/usr/bin/env bash
# profile.sh — Phase 3 counter collection (plan §4, §7). SEPARATE from timing.
# Targeted metric set (NOT --set full, which inflated the paper's Table V). Emits
# counters.jsonl. Metric names are CANDIDATES — resolve them on-box first:
#   ncu --query-metrics | grep -E 'dram__cycles_active|lts__t_sector_hit'
#
# WSL2 prerequisite: GPU performance counters must be enabled on the Windows host
# (NVIDIA Control Panel -> Developer -> Manage GPU Performance Counters ->
# "Allow access to all users"), else ncu fails with ERR_NVGPUCTRPERM.
set -euo pipefail

BENCH=${1:-./bench}
SIZES=${2:-512,1024,2048,4096,8192}
OUT=${3:-counters.jsonl}

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
# One kernel replay per launch; --launch-count keeps replays bounded. Parse the
# CSV ncu emits into JSONL. (Kept as a documented pipeline; the parser maps the
# raw metric names to the short keys analyze.py/check_consistency.py expect.)
ncu --metrics "$METRIC_CSV" --csv --target-processes all \
    --log-file ncu_raw.csv \
    "$BENCH" --sizes "$SIZES" --iters 1 --warmup 0 >/dev/null 2>&1 || true

python3 - "$OUT" <<'PY'
import csv, json, sys, re
out = sys.argv[1]
KEYMAP = {
 "dram__cycles_active.avg": "dram_cycles_active",
 "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_throughput_pct",
 "lts__t_sector_hit_rate.pct": "l2_hit_pct",
 "l1tex__t_sector_hit_rate.pct": "l1tex_hit_pct",
 "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio": "sectors_per_req",
 "smsp__thread_inst_executed_per_inst_executed.ratio": "active_threads_per_warp",
 "smsp__sass_branch_targets_threads_divergent.sum": "divergent_branches",
 "smsp__inst_executed.sum": "inst_executed",
 "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy_pct",
 "gpu__time_duration.sum": "ncu_duration_ms",
}
rows = {}
try:
    with open("ncu_raw.csv") as f:
        # ncu csv has a preamble; find the header line with 'Kernel Name'
        lines = [l for l in f if l.strip()]
    hdr_i = next(i for i,l in enumerate(lines) if "Metric Name" in l)
    rdr = csv.DictReader(lines[hdr_i:])
    for r in rdr:
        kn = r.get("Kernel Name","")
        m = r.get("Metric Name",""); v = r.get("Metric Value","")
        key = (kn,)  # NOTE: on-box, also parse the grid/N from demangled name
        rows.setdefault(key, {"kernel": kn})
        if m in KEYMAP:
            try: val=float(str(v).replace(",",""))
            except ValueError: val=v
            if KEYMAP[m]=="ncu_duration_ms": val=val/1e6  # ns->ms if reported in ns
            rows[key][KEYMAP[m]] = val
    with open(out,"w") as w:
        for v in rows.values(): w.write(json.dumps(v)+"\n")
    print(f"wrote {out} ({len(rows)} kernels)")
except (StopIteration, FileNotFoundError) as e:
    print(f"WARN: could not parse ncu_raw.csv ({e}); inspect it manually", file=sys.stderr)
PY
