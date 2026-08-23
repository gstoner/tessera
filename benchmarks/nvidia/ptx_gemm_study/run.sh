#!/usr/bin/env bash
# run.sh — end-to-end driver for SuperBear. Ordered so the pivotal INT4 question
# is answered first (Phase 0) and a red consistency gate blocks the report.
set -euo pipefail
cd "$(dirname "$0")"

SIZES=${SIZES:-512,1024,2048,4096,8192}
ITERS=${ITERS:-200}
WARMUP=${WARMUP:-20}
BATCH=${BATCH:-10}

echo "== archcheck =="
make archcheck

echo "== build =="
# fall back to NO_INT4 build if the s4 MMA is rejected at compile time
make all || make all NO_INT4=1

echo "== Phase 0: capability probe (INT4 pivotal) =="
./probe > capability_matrix.json
# Derive the --enable set: fail-closed. A family that uses a probed mma.sync
# instruction is enabled ONLY if that instruction is native_ok on this silicon;
# otherwise it is dropped (never launched) so an illegal instruction cannot abort
# the run. WMMA baselines do not use the probed inline asm and stay enabled.
ENABLE=$(python3 - <<'PY'
import json, sys
m = json.load(open("capability_matrix.json"))
sys.stderr.write(f"device={m['device']} cc={m['cc']} l2={m['l2_bytes']}\n")
status = {v["variant"]: v["status"] for v in m["variants"]}
for v in m["variants"]:
    sys.stderr.write(f"  {v['variant']:20s} {v['status']:16s} err={v['max_abs_err']}\n")
# families independent of the mma.sync probe (reliable WMMA / library paths)
enabled = ["fp16_wmma", "cublaslt_fp16", "cublaslt_int8"]
# probe-gated mma.sync families: variant -> study kernel name
GATED = {
    "int8_m16n8k32": "int8_ptx_mma_k32",
    "int4_m16n8k64": "int4_ptx_mma_k64,int4_wmma_preexpanded_fp16,int4_ptx_3stage",
}
for var, fam in GATED.items():
    if status.get(var) == "native_ok":
        enabled.append(fam)
    else:
        sys.stderr.write(f"  >>> DROPPED {fam}: probe says {var}={status.get(var,'absent')} "
                         f"(not native_ok) — fail-closed, remaining families still run\n")
if "int4_m16n8k64" in status:
    sys.stderr.write(f"  >>> PIVOTAL: native s4 on sm_120 => {status['int4_m16n8k64']}\n")
print(",".join(enabled))
PY
)
echo "enabled families: $ENABLE"

echo "== clock lock attempt (WSL2 may reject) =="
# Claim clocks_locked ONLY if BOTH graphics and memory clocks lock — a locked
# core clock with a floating memory clock still leaves bandwidth-sensitive
# measurements uncontrolled (plan §C7). Either failure -> rely on the CoV gate.
LOCKED=""
if sudo nvidia-smi -lgc 2100 >/dev/null 2>&1 && sudo nvidia-smi -lmc 9000 >/dev/null 2>&1; then
  LOCKED="--clocks-locked"; echo "graphics + memory clocks locked"
else
  sudo nvidia-smi -rgc >/dev/null 2>&1 || true   # undo a partial core-only lock
  sudo nvidia-smi -rmc >/dev/null 2>&1 || true
  echo "clocks NOT fully locked -> CoV gate carries stability (clocks_locked:false)"
fi

echo "== Phase 1+2: correctness + clean timing =="
./bench --sizes "$SIZES" --iters "$ITERS" --warmup "$WARMUP" --batch "$BATCH" --enable "$ENABLE" $LOCKED > results.jsonl
grep -c '"status":"OK"' results.jsonl && echo "OK rows above"

echo "== Phase 3: profiling (separate pass) =="
BATCH="$BATCH" bash profile.sh ./bench "$SIZES" counters.jsonl "$ENABLE" || echo "profiling skipped/failed (see stderr)"

echo "== Phase 4: consistency gate (blocks report if red) =="
python3 check_consistency.py results.jsonl counters.jsonl

echo "== Phase 4: report =="
python3 analyze.py results.jsonl counters.jsonl -o REPORT.md
echo "== Phase 4: freeze local native selector observation =="
python3 record_selector_decision.py results.jsonl selector_observation.json
echo "== Phase 5: guarded Tessera proposal =="
python3 phase5_ingest.py results.jsonl counters.jsonl capability_matrix.json \
  --output phase5_proposal.json
echo "done -> REPORT.md, results.jsonl, counters.jsonl, capability_matrix.json, selector_observation.json, phase5_proposal.json"
