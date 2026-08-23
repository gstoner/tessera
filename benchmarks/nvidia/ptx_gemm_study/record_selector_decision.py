#!/usr/bin/env python3
"""Freeze the study-local INT4 selector observation without promoting Tessera.

The output is evidence only.  It never edits ``mma_selector.py``: WSL evidence
cannot select a production schedule, and a later bare-metal packet must repeat
the comparison before a compiler policy changes.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

DIRECT = "int4_ptx_mma_k64"
STAGED = "int4_ptx_3stage"


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: record_selector_decision.py RESULTS.jsonl OUT.json", file=sys.stderr)
        return 2
    rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line]
    values = {(row.get("kernel"), row.get("n")): row for row in rows
              if row.get("status") == "OK" and row.get("timing_scope", "kernel") == "kernel"}
    decisions = []
    for n in sorted({n for (_, n) in values}):
        direct, staged = values.get((DIRECT, n)), values.get((STAGED, n))
        if not direct or not staged:
            continue
        winner = DIRECT if direct["latency_ms"] <= staged["latency_ms"] else STAGED
        decisions.append({"n": n, "winner": winner,
                          "direct_ms": direct["latency_ms"], "staged_ms": staged["latency_ms"]})
    if not decisions:
        print("no comparable native INT4 rows", file=sys.stderr)
        return 1
    Path(sys.argv[2]).write_text(json.dumps({
        "kind": "study_local_selector_observation",
        "target": "nvidia_sm120",
        "selector_eligible": False,
        "reason": "WSL evidence is retained only; no mma_selector.py update.",
        "decisions": decisions,
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
