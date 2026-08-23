#!/usr/bin/env bash
# Profile a cuBLASLt floor without conflating public-call and kernel durations.
#
# CUDA events surround the complete cuBLASLt call (dispatch plus the selected
# kernel).  NCU attaches to that selected internal kernel.  Both observations
# are valuable, but neither is substituted for the other in selector evidence.
# This protocol deliberately runs the library candidate by itself and records
# the two timing scopes in one small packet.
#
# Usage: profile_library_floor.sh fp16|int8 [N] [OUT.json]
set -euo pipefail
cd "$(dirname "$0")"

FAMILY=${1:-}
N=${2:-512}
OUT=${3:-"library_floor_${FAMILY}_${N}.json"}
WARMUP=${WARMUP:-50}
ITERS=${ITERS:-20}

case "$FAMILY" in
  fp16)
    ENABLE=cublaslt_fp16
    STUDY_KERNEL=cublaslt_fp16
    # CUDA 13.3 SM120 variants observed on SuperBear.  Keep the filtering
    # explicit so an unrelated kernel cannot be mistaken for the floor.
    KERNEL_FILTER='regex:(nvjet_sm120.*|.*cutlass_80_tensorop_s16816gemm_f16.*)'
    ;;
  int8)
    ENABLE=cublaslt_int8
    STUDY_KERNEL=cublaslt_int8
    KERNEL_FILTER='regex:nvjet_sm120_bii.*'
    ;;
  *)
    echo "usage: $0 fp16|int8 [N] [OUT.json]" >&2
    exit 2
    ;;
esac

if ! ncu --query-metrics >/dev/null 2>&1; then
  echo "NCU counter access is unavailable (ERR_NVGPUCTRPERM?): cannot profile $FAMILY" >&2
  exit 1
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# A single call establishes the public-call timing floor.  The batched run
# amortizes event-recording noise but remains a public-call measurement.
./bench --sizes "$N" --exact-max-n "$N" --iters "$ITERS" --warmup "$WARMUP" \
  --batch 1 --enable "$ENABLE" > "$TMP/events_single.jsonl"
./bench --sizes "$N" --exact-max-n "$N" --iters "$ITERS" --warmup "$WARMUP" \
  --batch 100 --enable "$ENABLE" > "$TMP/events_batched.jsonl"

# One correctness call precedes warmup calls in bench.  Skip it plus warmups,
# then collect one selected library kernel.  This is intentionally a separate
# process from event timing.
ncu --metrics gpu__time_duration.sum --csv --target-processes all \
  --kernel-name-base demangled --kernel-name "$KERNEL_FILTER" \
  --launch-skip "$((WARMUP + 1))" --launch-count 1 --log-file "$TMP/ncu.csv" \
  ./bench --sizes "$N" --exact-max-n "$N" --iters 1 --warmup "$WARMUP" \
  --batch 1 --enable "$ENABLE" >/dev/null
python3 parse_ncu.py "$TMP/ncu.csv" "$N" > "$TMP/ncu.jsonl"

python3 - "$TMP/events_single.jsonl" "$TMP/events_batched.jsonl" \
  "$TMP/ncu.jsonl" "$FAMILY" "$STUDY_KERNEL" "$N" "$OUT" <<'PY'
import json
import pathlib
import sys

single_path = pathlib.Path(sys.argv[1])
batched_path = pathlib.Path(sys.argv[2])
ncu_path = pathlib.Path(sys.argv[3])
family = sys.argv[4]
study_kernel = sys.argv[5]
n = int(sys.argv[6])
out = pathlib.Path(sys.argv[7])

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]

def one(path):
    found = [r for r in rows(path) if r.get("kernel") == study_kernel and r.get("status") == "OK"]
    if len(found) != 1:
        raise SystemExit(f"expected exactly one {family} row in {path}, found {len(found)}")
    return found[0]

event_single = one(single_path)
event_batched = one(batched_path)
ncu = [r for r in rows(ncu_path) if r.get("kernel") == study_kernel and r.get("ncu_duration_ms") is not None]
if len(ncu) != 1:
    raise SystemExit(f"expected exactly one profiled {family} kernel, found {len(ncu)}")

pathlib.Path(out).write_text(json.dumps({
    "kind": "cublaslt_floor_timing_protocol",
    "family": family,
    "n": n,
    "selector_eligible": False,
    "reason": "WSL library-call and internal-kernel timing scopes are not interchangeable.",
    "event_single_call": {"timing_scope": event_single.get("timing_scope"),
                          "latency_ms": event_single["latency_ms"], "cov": event_single["cov"]},
    "event_batched_call": {"batch": 100, "timing_scope": event_batched.get("timing_scope"),
                             "latency_ms": event_batched["latency_ms"], "cov": event_batched["cov"]},
    "ncu_selected_kernel": {"profiling_scope": ncu[0].get("profiling_scope"),
                              "duration_ms": ncu[0]["ncu_duration_ms"]},
}, indent=2) + "\n")
PY

echo "wrote $OUT"
