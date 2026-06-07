"""Run the lattice reasoning core benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "python"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benchmarks.lattice_reasoning_core import build_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Use the tiny 4x4x4 lattice shape")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--json", type=str, default=None, help="Optional output JSON path")
    args = parser.parse_args(argv)

    report = build_report(smoke=args.smoke, reps=args.reps, seed=args.seed)
    text = json.dumps(report, indent=2)
    if args.json:
        Path(args.json).write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
