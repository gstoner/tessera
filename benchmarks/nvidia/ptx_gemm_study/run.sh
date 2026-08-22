#!/usr/bin/env bash
# run.sh — end-to-end driver for SuperBear. Ordered so the pivotal INT4 question
# is answered first (Phase 0) and a red consistency gate blocks the report.
set -euo pipefail
cd "$(dirname "$0")"

SIZES=${SIZES:-512,1024,2048,4096,8192}
ITERS=${ITERS:-200}
WARMUP=${WARMUP:-20}

echo "== archcheck =="
make archcheck

echo "== build =="
# fall back to NO_INT4 build if the s4 MMA is rejected at compile time
make all || make all NO_INT4=1

echo "== Phase 0: capability probe (INT4 pivotal) =="
./probe > capability_matrix.json
python3 - <<'PY'
import json
m=json.load(open("capability_matrix.json"))
print(f"device={m['device']} cc={m['cc']} l2={m['l2_bytes']}")
for v in m["variants"]:
    print(f"  {v['variant']:20s} {v['status']:16s} err={v['max_abs_err']}")
    if v["variant"].startswith("int4"):
        print(f"  >>> PIVOTAL: native s4 on sm_120 => {v['status']}")
PY

echo "== clock lock attempt (WSL2 may reject) =="
LOCKED=""
if sudo nvidia-smi -lgc 2100 >/dev/null 2>&1; then LOCKED="--clocks-locked"; echo "locked"; else echo "not locked -> CoV gate carries stability"; fi

echo "== Phase 1+2: correctness + clean timing =="
./bench --sizes "$SIZES" --iters "$ITERS" --warmup "$WARMUP" $LOCKED > results.jsonl
grep -c '"status":"OK"' results.jsonl && echo "OK rows above"

echo "== Phase 3: profiling (separate pass) =="
bash profile.sh ./bench "$SIZES" counters.jsonl || echo "profiling skipped/failed (see stderr)"

echo "== Phase 4: consistency gate (blocks report if red) =="
python3 check_consistency.py results.jsonl counters.jsonl

echo "== Phase 4: report =="
python3 analyze.py results.jsonl counters.jsonl -o REPORT.md
echo "done -> REPORT.md, results.jsonl, counters.jsonl, capability_matrix.json"
