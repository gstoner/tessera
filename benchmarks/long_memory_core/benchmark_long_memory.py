"""Run the long-horizon memory core benchmark.

Emits a proof-level summary (and optionally the per-row + telemetry JSON) for the
RULER / LongMemEval / MemoryArena-style scenarios.  See ``core.py`` for the
promotion-ladder discipline: reference oracle first, missing-backend rows name
the open gaps in :data:`MEMORY_PRIMITIVE_GAPS`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "python"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benchmarks.long_memory_core import (
    LongMemoryConfig,
    adapter_report,
    build_report,
    run_all_adapters,
    run_core,
    telemetry,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny bank (32 entries) for a fast smoke run")
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--rows", action="store_true",
                        help="Include the per-row flat dicts in the output")
    parser.add_argument("--telemetry", action="store_true",
                        help="Include the shared Tessera telemetry events")
    parser.add_argument("--adapters", action="store_true",
                        help="Also run the LongMemEval/MemoryArena/LongBench-v2 adapters")
    parser.add_argument("--json", type=str, default=None, help="Optional output JSON path")
    args = parser.parse_args(argv)

    cfg = LongMemoryConfig(bank_size=32 if args.smoke else 256, seed=args.seed)
    rows = run_core(cfg)
    out: dict[str, object] = {"report": build_report(rows)}
    if args.adapters:
        out["adapters"] = adapter_report(run_all_adapters(seed=args.seed))
    if args.rows:
        out["rows"] = [r.flat_dict() for r in rows]
    if args.telemetry:
        out["telemetry"] = telemetry(rows)

    text = json.dumps(out, indent=2)
    if args.json:
        Path(args.json).write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
