"""Aggregate independent Apple route reports into a promotion ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tessera.compiler.apple_route_selector import aggregate_stable_route_reports


def _incumbent(spec: str) -> tuple[str, str]:
    op, separator, route = spec.partition("=")
    if not separator or not op or not route:
        raise argparse.ArgumentTypeError("incumbent route must be OP=ROUTE")
    return op, route


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--minimum-speedup", type=float, default=0.05)
    parser.add_argument("--maximum-run-drift", type=float, default=0.15)
    parser.add_argument("--minimum-paired-win-fraction", type=float, default=0.75)
    parser.add_argument("--maximum-speedup-spread", type=float, default=0.05)
    parser.add_argument(
        "--incumbent-route", action="append", type=_incumbent, default=[],
        metavar="OP=ROUTE",
        help="Override the production incumbent for an operation (repeatable)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.reports) < 2:
        parser.error("at least two independent reports are required")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    ledger = aggregate_stable_route_reports(
        reports,
        incumbent_routes=dict(args.incumbent_route),
        min_speedup=args.minimum_speedup,
        max_run_drift=args.maximum_run_drift,
        min_paired_win_fraction=args.minimum_paired_win_fraction,
        max_speedup_spread=args.maximum_speedup_spread,
    )
    args.output.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
