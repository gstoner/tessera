"""Run the DLOP-Bench-style long-tail operator fusion benchmark.

Emits per-composite dispatch-count / decomposition-factor / fusion-reduction
rows (fused ≡ eager-decomposed, metamorphically gated). See core.py.
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

from benchmarks.dlop_longtail_core import build_report, run_core, telemetry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--rows", action="store_true", help="Include per-row dicts")
    parser.add_argument("--telemetry", action="store_true", help="Include telemetry events")
    parser.add_argument("--json", type=str, default=None, help="Optional output JSON path")
    args = parser.parse_args(argv)

    rows = run_core()
    out: dict[str, object] = {"report": build_report(rows)}
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
