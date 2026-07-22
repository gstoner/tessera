"""Seal independently captured Apple route reports into strict-v2 evidence.

This tool never converts a schema-v1 ledger.  Its inputs are the raw reports
from an owning benchmark path, so their exact Apple context, oracle result,
resource records, paired trials, and source digests remain reviewable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tessera.compiler.apple_route_selector import (
    STRICT_PACKAGE_SUBGRAPH_SCOPE,
    STRICT_RUNTIME_ROUTE_SCOPE,
    aggregate_stable_route_reports,
    seal_strict_route_ledger,
)


def _incumbent(spec: str) -> tuple[str, str]:
    op, separator, route = spec.partition("=")
    if not separator or not op or not route:
        raise argparse.ArgumentTypeError("incumbent route must be OP=ROUTE")
    return op, route


def seal_reports(
    reports: list[dict[str, object]], *, incumbents: dict[str, str],
    valid_days: int = 30, selection_scope: str = STRICT_RUNTIME_ROUTE_SCOPE,
) -> dict[str, object]:
    """Aggregate and seal reports produced by one Apple benchmark family."""
    stable = aggregate_stable_route_reports(reports, incumbent_routes=incumbents)
    return seal_strict_route_ledger(
        stable, reports, valid_days=valid_days, selection_scope=selection_scope,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--incumbent-route", action="append", type=_incumbent,
                        default=[], metavar="OP=ROUTE")
    parser.add_argument("--valid-days", type=int, default=30)
    parser.add_argument(
        "--selection-scope",
        choices=(STRICT_RUNTIME_ROUTE_SCOPE, STRICT_PACKAGE_SUBGRAPH_SCOPE),
        default=STRICT_RUNTIME_ROUTE_SCOPE,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.reports) < 2:
        parser.error("at least two independently collected reports are required")
    if args.valid_days < 1:
        parser.error("--valid-days must be positive")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    ledger = seal_reports(reports, incumbents=dict(args.incumbent_route),
                          valid_days=args.valid_days,
                          selection_scope=args.selection_scope)
    args.output.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
